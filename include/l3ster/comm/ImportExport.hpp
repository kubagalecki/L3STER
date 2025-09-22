#ifndef L3STER_COMM_IMPORTEXPORT_HPP
#define L3STER_COMM_IMPORTEXPORT_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/CrsGraph.hpp"
#include "l3ster/util/IndexMap.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"
#include "l3ster/util/SegmentedOwnership.hpp"
#include "l3ster/util/TbbUtils.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

// Some general notes:
// The import/export classes below essentially perform similar tasks to Tpetra::Import/Export, with the following
// differences:
// - more granular interface: sends and receives (and export unpacks) are handled individually
// - thread safety: members starting with try/test can be called concurrently (although MPI limitations still apply)
// - stronger preconditions for input maps (segmented ownership): some comms are sent/received directly without packing

namespace lstr::comm
{
/// Describes data movement between owned and shared indices
/// This is expensive to construct
/// Conventions:
/// - owned map -> which ranks share my owned inds (how to pack for import sends)
/// - shared map -> which ranks own my shared inds (how to unpack export receives)
/// - local indexing of owned and shared inds is separate
class ImportExportContext
{
    using LocalIndex = local_dof_t;

public:
    using NbrSpan = std::span< const int >;
    using IndSpan = std::span< const LocalIndex >;
    struct SharedInds
    {
        int        rank;
        LocalIndex offset;
        size_t     size;
    };

    template < std::integral GlobalIndex >
    ImportExportContext(const MpiComm& comm, const util::SegmentedOwnership< GlobalIndex >& ownership);

    auto getNumNbrs() const -> size_t { return m_nbrs.size(); }
    auto getOwnedNbrs() const -> NbrSpan { return {m_nbrs | std::views::take(m_num_owned_nbrs)}; }
    auto getOwnedInds() const -> IndSpan { return m_owned_inds(0, m_num_owned_nbrs); }
    auto getOwnedRange() const
    {
        return std::views::iota(0uz, m_num_owned_nbrs) |
               std::views::transform([this](size_t i) { return std::make_pair(m_nbrs.at(i), m_owned_inds(i)); });
    }
    auto getSharedNbrs() const -> NbrSpan { return {m_nbrs | std::views::drop(m_num_owned_nbrs)}; }
    auto getNumSharedInds() const -> size_t { return static_cast< size_t >(m_shared_ind_offsets.back()); }
    auto getSharedRange() const
    {
        return std::views::zip_transform(
            [](int rank, LocalIndex offset, LocalIndex offset_next) {
                return SharedInds{rank, offset, static_cast< size_t >(offset_next - offset)};
            },
            getSharedNbrs(),
            m_shared_ind_offsets,
            m_shared_ind_offsets | std::views::drop(1));
    }

private:
    size_t                         m_num_owned_nbrs;
    util::ArrayOwner< int >        m_nbrs;
    util::CrsGraph< LocalIndex >   m_owned_inds;
    util::ArrayOwner< LocalIndex > m_shared_ind_offsets;
};

template < Arithmetic_c Scalar >
class ImportExportBase
{
    using LocalIndex = local_dof_t;

    // Each import/export object has a unique ID, which is used to deconflict comms from different objects
    inline static unsigned short id_counter   = 1;
    static constexpr size_t      tag_bits     = 12;
    static constexpr size_t      id_bits      = 8;
    static constexpr size_t      max_num_vecs = 1uz << tag_bits;
    static constexpr int         int_bits     = sizeof(int) * CHAR_BIT;
    static constexpr int         tag_mask     = -1u >> (int_bits - tag_bits);
    static constexpr int         id_mask      = -1u >> (int_bits - id_bits);

public:
    auto                 getContext() const { return m_context; }
    [[nodiscard]] size_t getNumVecs() const { return m_num_vecs; }
    void                 setNumVecs(size_t num_vecs)
    {
        util::throwingAssert(num_vecs <= m_num_vecs_max);
        m_num_vecs = num_vecs;
    }

protected:
    [[nodiscard]] int makeTag(int tag) const { return (tag & tag_mask) | std::bit_cast< int >(m_id); }

    using context_shared_ptr_t = std::shared_ptr< const ImportExportContext >;

    ImportExportBase(context_shared_ptr_t context, size_t num_vecs)
        : m_context{std::move(context)},
          m_num_vecs{num_vecs},
          m_num_vecs_max{num_vecs},
          m_requests(num_vecs * m_context->getNumNbrs()),
          m_pack_buf(num_vecs * m_context->getOwnedInds().size()),
          m_max_owned{m_context->getOwnedInds().empty() ? -1 : std::ranges::max(m_context->getOwnedInds())},
          m_id{(id_mask & id_counter++) << tag_bits}
    {
        util::throwingAssert(num_vecs < max_num_vecs);
        util::throwingAssert(makeTag(-1) <= getMaxMpiTag(), "MPI_TAG_UB value too low");
    }

    bool isOwnedSizeSufficient(size_t size) const;
    bool isSharedSizeSufficient(size_t size) const;

    std::shared_ptr< const ImportExportContext > m_context{};
    size_t                                       m_num_vecs{};
    size_t                                       m_num_vecs_max{};
    util::ArrayOwner< MpiComm::Request >         m_requests;
    util::ArrayOwner< Scalar >                   m_pack_buf;
    size_t                                       m_owned_stride{}, m_shared_stride{};
    LocalIndex                                   m_max_owned{};
    const int                                    m_id;
};

/// Performs import, i.e., data at shared indices is received from owning ranks
/// Cheap to construct (allocation only) and reuse, since it uses an existing context
template < Arithmetic_c Scalar >
class Import : public ImportExportBase< Scalar >
{
    using Base = ImportExportBase< Scalar >;

public:
    Import(Base::context_shared_ptr_t context, size_t num_vecs) : Base{std::move(context), num_vecs} {}

    inline void setOwned(std::span< const Scalar > owned_vals, size_t owned_stride);
    inline void setShared(std::span< Scalar > shared_vals, size_t shared_stride);
    inline void postComms(const MpiComm& comm);
    inline bool tryReceive(); // Returns true only once, can be called concurrently
    bool        testReceive() const { return m_recv_complete.test(std::memory_order_acquire); }
    void        waitReceive() { MpiComm::Request::waitAll(getRecvRequests()); }
    void        wait() { MpiComm::Request::waitAll(Base::m_requests); }
    void        doBlockingImport(const MpiComm& comm)
    {
        postComms(comm);
        wait();
    }

    void setOwned(const Kokkos::View< const Scalar**, Kokkos::LayoutLeft >& view)
    {
        setOwned(util::getMemorySpan(view), view.stride(1));
    }
    void setShared(const Kokkos::View< Scalar**, Kokkos::LayoutLeft >& view)
    {
        setShared(util::getMemorySpan(view), view.stride(1));
    }

private:
    inline void postSends(const MpiComm& comm);
    inline void postRecvs(const MpiComm& comm);
    inline auto getSendRequests() -> std::span< MpiComm::Request >;
    inline auto getRecvRequests() -> std::span< MpiComm::Request >;
    inline void packSends();

    const Scalar*    m_owned  = nullptr;
    Scalar*          m_shared = nullptr;
    std::atomic_flag m_recv_complete;
    std::mutex       m_mutex;
};

/// Performs export, i.e., data at shared indices is sent to owning ranks
/// Cheap to construct (allocation only) and reuse, since it uses an existing context
template < Arithmetic_c Scalar >
class Export : public ImportExportBase< Scalar >
{
    using Base = ImportExportBase< Scalar >;

public:
    Export(Base::context_shared_ptr_t context, size_t num_vecs) : Base{std::move(context), num_vecs} {}

    inline void setOwned(std::span< Scalar > owned_vals, size_t owned_stride);
    inline void setShared(std::span< const Scalar > shared_vals, size_t shared_stride);
    inline void postSends(const MpiComm& comm);
    inline void postRecvs(const MpiComm& comm);
    template < std::invocable< Scalar&, Scalar > Combine >
    void wait(Combine&& combine);
    template < std::invocable< Scalar&, Scalar > Combine >
    void doBlockingExport(const MpiComm& comm, Combine&& combine)
    {
        postRecvs(comm);
        postSends(comm);
        wait(std::forward< Combine >(combine));
    }

    void setOwned(const Kokkos::View< Scalar**, Kokkos::LayoutLeft >& view)
    {
        setOwned(util::getMemorySpan(view), view.stride(1));
    }
    void setShared(const Kokkos::View< const Scalar**, Kokkos::LayoutLeft >& view)
    {
        setShared(util::getMemorySpan(view), view.stride(1));
    }

private:
    inline auto getSendRequests() -> std::span< MpiComm::Request >;
    inline auto getRecvRequests() -> std::span< MpiComm::Request >;

    template < std::invocable< Scalar&, Scalar > Combine >
    void unpack(Combine&& combine);

    Scalar*       m_owned  = nullptr;
    const Scalar* m_shared = nullptr;
};

namespace detail
{
template < typename GlobalIndex >
auto getNeighbors(const std::map< int, std::vector< GlobalIndex > >& out_nbr_info,
                  const std::map< int, std::vector< GlobalIndex > >& in_nbr_info) -> util::ArrayOwner< int >
{
    return {std::array{in_nbr_info | std::views::keys, out_nbr_info | std::views::keys} | std::views::join};
}

template < typename LocalIndex, typename GlobalIndex >
auto flattenOwned(const util::SegmentedOwnership< GlobalIndex >&     ownership,
                  const std::map< int, std::vector< GlobalIndex > >& in_nbr_info) -> util::CrsGraph< LocalIndex >
{
    const auto g2l    = std::bind_front(&util::SegmentedOwnership< GlobalIndex >::getLocalIndex, std::cref(ownership));
    auto       retval = util::CrsGraph< LocalIndex >(in_nbr_info |
                                               std::views::transform([](const auto& p) { return p.second.size(); }));
    for (auto&& [i, entries] : in_nbr_info | std::views::values | std::views::enumerate)
        std::ranges::transform(entries, retval(i).begin(), g2l);
    return retval;
}

template < typename LocalIndex, typename GlobalIndex >
auto getOffsets(const std::map< int, std::vector< GlobalIndex > >& out_nbr_info) -> util::ArrayOwner< LocalIndex >
{
    auto retval           = util::ArrayOwner< LocalIndex >(out_nbr_info.size() + 1);
    retval.front()        = 0;
    const auto sizes_view = out_nbr_info | std::views::values |
                            std::views::transform(&std::vector< GlobalIndex >::size) | std::views::common;
    std::inclusive_scan(sizes_view.begin(), sizes_view.end(), std::next(retval.begin()));
    return retval;
}
} // namespace detail

template < std::integral GlobalIndex >
ImportExportContext::ImportExportContext(const MpiComm& comm, const util::SegmentedOwnership< GlobalIndex >& ownership)
{
    L3STER_PROFILE_FUNCTION;
    const auto own_dist     = ownership.getOwnershipDist(comm);
    const auto out_nbr_info = ownership.computeOutNbrInfo(own_dist);
    const auto in_nbr_info  = ownership.computeInNbrInfo(comm, out_nbr_info);
    m_nbrs                  = detail::getNeighbors(out_nbr_info, in_nbr_info);
    m_num_owned_nbrs        = in_nbr_info.size();
    m_owned_inds            = detail::flattenOwned< LocalIndex >(ownership, in_nbr_info);
    m_shared_ind_offsets    = detail::getOffsets< LocalIndex >(out_nbr_info);
}

template < Arithmetic_c Scalar >
bool ImportExportBase< Scalar >::isOwnedSizeSufficient(size_t size) const
{
    const auto num_owned     = static_cast< size_t >(m_max_owned + 1);
    const auto required_size = (m_num_vecs - 1) * m_owned_stride + num_owned;
    return num_owned > 0 ? size >= required_size : true;
}

template < Arithmetic_c Scalar >
bool ImportExportBase< Scalar >::isSharedSizeSufficient(size_t size) const
{
    const auto num_shared    = m_context->getNumSharedInds();
    const auto required_size = (m_num_vecs - 1) * m_shared_stride + num_shared;
    return num_shared > 0 ? size >= required_size : true;
}

template < Arithmetic_c Scalar >
void Import< Scalar >::setOwned(std::span< const Scalar > owned_vals, size_t owned_stride)
{
    Base::m_owned_stride = owned_stride;
    util::throwingAssert(Base::isOwnedSizeSufficient(owned_vals.size()));
    m_owned = owned_vals.data();
}

template < Arithmetic_c Scalar >
void Import< Scalar >::setShared(std::span< Scalar > shared_vals, size_t shared_stride)
{
    Base::m_shared_stride = shared_stride;
    util::throwingAssert(Base::isSharedSizeSufficient(shared_vals.size()));
    m_shared = shared_vals.data();
}

template < Arithmetic_c Scalar >
void Import< Scalar >::postComms(const MpiComm& comm)
{
    util::throwingAssert(m_owned, "`setOwned` must be called before `postComms`");
    util::throwingAssert(m_shared, "`setShared` must be called before `postComms`");
    postRecvs(comm);
    postSends(comm);
    m_recv_complete.clear(std::memory_order_release);
}

template < Arithmetic_c Scalar >
void Import< Scalar >::postRecvs(const MpiComm& comm)
{
    const auto req_span = getRecvRequests();
    size_t     req_ind  = 0;
    for (size_t vec = 0; vec != Base::m_num_vecs; ++vec)
    {
        const auto shared_vec = std::span{std::next(m_shared, vec * Base::m_shared_stride), Base::m_shared_stride};
        for (const auto& [src_rank, inds_offset, inds_size] : Base::m_context->getSharedRange())
        {
            const auto recv_span = shared_vec.subspan(inds_offset, inds_size);
            const int  tag       = Base::makeTag(static_cast< int >(vec));
            req_span[req_ind++]  = comm.receiveAsync(recv_span, src_rank, tag);
        }
    }
}

template < Arithmetic_c Scalar >
void Import< Scalar >::postSends(const MpiComm& comm)
{
    packSends();
    const auto buf_span  = std::span{std::as_const(Base::m_pack_buf)};
    const auto req_span  = getSendRequests();
    size_t     buf_index = 0, i = 0;
    for (size_t vec = 0; vec != Base::m_num_vecs; ++vec)
        for (const auto& [dest_rank, inds] : Base::m_context->getOwnedRange())
        {
            const auto send_data = buf_span.subspan(buf_index, inds.size());
            const int  tag       = Base::makeTag(static_cast< int >(vec));
            req_span[i++]        = comm.sendAsync(send_data, dest_rank, tag);
            buf_index += send_data.size();
        }
}

template < Arithmetic_c Scalar >
bool Import< Scalar >::tryReceive()
{
    // Double-checked locking
    if (m_recv_complete.test(std::memory_order_acquire))
        return false;
    auto lock = std::unique_lock{m_mutex, std::defer_lock};
    if (not lock.try_lock())
        return false;
    if (m_recv_complete.test(std::memory_order_relaxed))
        return false;
    const auto completed = std::ranges::all_of(getRecvRequests(), &MpiComm::Request::test);
    if (completed)
        m_recv_complete.test_and_set(std::memory_order_release);
    return completed;
}

template < Arithmetic_c Scalar >
void Import< Scalar >::packSends()
{
    const auto owned_inds = Base::m_context->getOwnedInds();
    const auto pack_vec   = [&](size_t vec) {
        const auto pack_stride = Base::m_context->getOwnedInds().size();
        const auto pack_offs   = vec * pack_stride;
        const auto dest_span   = std::span{Base::m_pack_buf}.subspan(pack_offs, pack_stride);
        const auto owned_offs  = vec * Base::m_owned_stride;
        const auto src_span    = std::span{std::next(m_owned, owned_offs), Base::m_owned_stride};
        const auto pack_ind    = [&](size_t i) {
            dest_span[i] = src_span[owned_inds[i]];
        };
        util::tbb::parallelFor(std::views::iota(0uz, owned_inds.size()), pack_ind);
    };
    util::tbb::parallelFor(std::views::iota(0uz, Base::m_num_vecs), pack_vec);
}

template < Arithmetic_c Scalar >
auto Import< Scalar >::getRecvRequests() -> std::span< MpiComm::Request >
{
    return {Base::m_requests | std::views::take(Base::m_context->getSharedNbrs().size() * Base::m_num_vecs)};
}

template < Arithmetic_c Scalar >
auto Import< Scalar >::getSendRequests() -> std::span< MpiComm::Request >
{
    return {Base::m_requests | std::views::drop(Base::m_context->getSharedNbrs().size() * Base::m_num_vecs)};
}

template < Arithmetic_c Scalar >
void Export< Scalar >::setOwned(std::span< Scalar > owned_vals, size_t owned_stride)
{
    Base::m_owned_stride = owned_stride;
    util::throwingAssert(Base::isOwnedSizeSufficient(owned_vals.size()));
    m_owned = owned_vals.data();
}

template < Arithmetic_c Scalar >
void Export< Scalar >::setShared(std::span< const Scalar > shared_vals, size_t shared_stride)
{
    Base::m_shared_stride = shared_stride;
    util::throwingAssert(Base::isSharedSizeSufficient(shared_vals.size()));
    m_shared = shared_vals.data();
}

template < Arithmetic_c Scalar >
void Export< Scalar >::postRecvs(const lstr::MpiComm& comm)
{
    util::throwingAssert(m_owned, "`setOwned` must be called before `postRecvs`");
    const auto recv_reqs = getRecvRequests();
    for (size_t offset = 0, req_ind = 0, vec = 0; vec != Base::m_num_vecs; ++vec)
        for (const auto& [rank, inds] : Base::m_context->getOwnedRange())
        {
            const auto recv_span = std::span{Base::m_pack_buf}.subspan(offset, inds.size());
            const int  tag       = Base::makeTag(static_cast< int >(vec));
            recv_reqs[req_ind++] = comm.receiveAsync(recv_span, rank, tag);
            offset += inds.size();
        }
}

template < Arithmetic_c Scalar >
void Export< Scalar >::postSends(const MpiComm& comm)
{
    util::throwingAssert(m_shared, "`setShared` must be called before `postSends`");
    const auto send_reqs     = getSendRequests();
    const auto shared_stride = Base::m_shared_stride;
    for (size_t req_ind = 0, vec = 0; vec != Base::m_num_vecs; ++vec)
    {
        const auto send_span = std::span{std::next(m_shared, vec * shared_stride), shared_stride};
        for (const auto& [rank, inds_offset, inds_size] : Base::m_context->getSharedRange())
        {
            const auto dest_data = send_span.subspan(inds_offset, inds_size);
            const int  tag       = Base::makeTag(static_cast< int >(vec));
            send_reqs[req_ind++] = comm.sendAsync(dest_data, rank, tag);
        }
    }
}

template < Arithmetic_c Scalar >
auto Export< Scalar >::getRecvRequests() -> std::span< MpiComm::Request >
{
    return {Base::m_requests | std::views::take(Base::m_context->getOwnedNbrs().size() * Base::m_num_vecs)};
}

template < Arithmetic_c Scalar >
auto Export< Scalar >::getSendRequests() -> std::span< MpiComm::Request >
{
    return {Base::m_requests | std::views::drop(Base::m_context->getOwnedNbrs().size() * Base::m_num_vecs)};
}

template < Arithmetic_c Scalar >
template < std::invocable< Scalar&, Scalar > Combine >
void Export< Scalar >::unpack(Combine&& combine)
{
    const auto owned_stride = Base::m_owned_stride;
    util::tbb::parallelFor(std::views::iota(0uz, Base::m_num_vecs), [&](size_t vec) {
        const auto src_offset = Base::m_context->getOwnedInds().size() * vec;
        const auto dest_offs  = vec * owned_stride;
        const auto dest_span  = std::span{std::next(m_owned, dest_offs), owned_stride};
        util::tbb::parallelFor(Base::m_context->getOwnedInds() | std::views::enumerate, [&](const auto& inds) {
            const auto& [i, owned] = inds;
            combine(dest_span[owned], Base::m_pack_buf[src_offset + i]);
        });
    });
}

template < Arithmetic_c Scalar >
template < std::invocable< Scalar&, Scalar > Combine >
void Export< Scalar >::wait(Combine&& combine)
{
    MpiComm::Request::waitAll(getRecvRequests());
    unpack(std::forward< Combine >(combine));
    MpiComm::Request::waitAll(getSendRequests());
}
} // namespace lstr::comm
#endif // L3STER_COMM_IMPORTEXPORT_HPP
