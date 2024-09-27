#ifndef L3STER_COMM_IMPORTEXPORT_HPP
#define L3STER_COMM_IMPORTEXPORT_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/CrsGraph.hpp"
#include "l3ster/util/IndexMap.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"
#include "l3ster/util/TbbUtils.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

// Some general notes:
// The import/export classes below essentially perform similar tasks to Tpetra::Import/Export, with the following
// differences:
// - more granular interface: sends and receives (and export unpacks) are handled individually
// - thread safety: members starting with try/test can be called concurrently (although MPI limitations still apply)
// - stronger preconditions for input maps (Amesos2 ordering): some comms are sent/received directly without packing

namespace lstr::comm
{
/// Describes data movement between owned and shared indices
/// This is expensive to construct
/// Conventions:
/// - owned map -> which ranks share my owned inds (how to pack for import sends)
/// - shared map -> which ranks own my shared inds (how to unpack export receives)
/// - local indexing of owned and shared inds is separate
template < std::signed_integral LocalIndex = std::int32_t >
class ImportExportContext
{
public:
    using NbrSpan = std::span< const int >;
    using IndSpan = std::span< const LocalIndex >;
    struct SharedInds
    {
        int        rank;
        LocalIndex offset;
        size_t     size;
    };

    template < std::signed_integral GlobalIndex >
    ImportExportContext(const MpiComm&                 comm,
                        std::span< const GlobalIndex > owned_inds,
                        std::span< const GlobalIndex > shared_inds);

    auto getNumNbrs() const -> size_t { return m_nbrs.size(); }
    auto getOwnedNbrs() const -> NbrSpan { return {m_nbrs | std::views::take(m_num_owned_nbrs)}; }
    auto getOwnedInds() const -> IndSpan { return m_owned_inds(0, m_num_owned_nbrs); }
    auto getOwnedRange() const
    {
        return std::views::iota(size_t{0}, m_num_owned_nbrs) |
               std::views::transform([this](size_t i) { return std::make_pair(m_nbrs.at(i), m_owned_inds(i)); });
    }
    auto getSharedNbrs() const -> NbrSpan { return {m_nbrs | std::views::drop(m_num_owned_nbrs)}; }
    auto getNumSharedInds() const -> size_t { return m_shared_ind_offsets.back(); }
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
template < std::ranges::contiguous_range R1, std::ranges::contiguous_range R2 >
ImportExportContext(const MpiComm& comm,
                    R1&&,
                    R2&&) -> ImportExportContext< std::remove_const_t< std::ranges::range_value_t< R1 > > >;

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
class ImportExportBase
{
    // Each import/export object has a unique ID, which is used to deconflict comms from different objects
    inline static unsigned counter   = 0;
    static constexpr int   half_bits = sizeof(int) * CHAR_BIT / 2;
    static constexpr int   lo_mask   = -1 >> half_bits;

public:
    auto getContext() const { return m_context; }

protected:
    int makeTag(int tag) { return (tag & lo_mask) | std::bit_cast< int >(m_id << half_bits); }

    using context_shared_ptr_t = std::shared_ptr< const ImportExportContext< LocalIndex > >;

    ImportExportBase(context_shared_ptr_t context, size_t num_vecs)
        : m_context{std::move(context)},
          m_num_vecs{num_vecs},
          m_requests(num_vecs * m_context->getNumNbrs()),
          m_pack_buf(num_vecs * m_context->getOwnedInds().size()),
          m_max_owned{m_context->getOwnedInds().empty() ? -1 : std::ranges::max(m_context->getOwnedInds())},
          m_id{counter++}
    {}

    bool isOwnedSizeSufficient(size_t size) const;
    bool isSharedSizeSufficient(size_t size) const;

    std::shared_ptr< const ImportExportContext< LocalIndex > > m_context{};
    size_t                                                     m_num_vecs{};
    util::ArrayOwner< MpiComm::Request >                       m_requests;
    util::ArrayOwner< Scalar >                                 m_pack_buf;
    size_t                                                     m_owned_stride{}, m_shared_stride{};
    LocalIndex                                                 m_max_owned{};
    const unsigned                                             m_id;
};

/// Performs import, i.e., data at shared indices is received from owning ranks
/// Cheap to construct (allocation only) and reuse, since it uses an existing context
template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
class Import : public ImportExportBase< Scalar, LocalIndex >
{
    using Base = ImportExportBase< Scalar, LocalIndex >;

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

    template < util::KokkosView_c View >
    void setOwned(View&& view)
        requires(std::remove_cvref_t< View >::rank() == 2)
    {
        setOwned(util::flatten(view), view.stride(0));
    }
    template < util::KokkosView_c View >
    void setShared(View&& view)
        requires(std::remove_cvref_t< View >::rank() == 2 and
                 not std::is_const_v< typename std::remove_cvref_t< View >::value_type >)
    {
        setShared(util::flatten(view), view.stride(0));
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
template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
class Export : public ImportExportBase< Scalar, LocalIndex >
{
    using Base = ImportExportBase< Scalar, LocalIndex >;

public:
    Export(Base::context_shared_ptr_t context, size_t num_vecs) : Base{std::move(context), num_vecs} {}

    inline void setOwned(std::span< Scalar > owned_vals, size_t owned_stride);
    inline void setShared(std::span< const Scalar > shared_vals, size_t shared_stride);
    inline void postSends(const MpiComm& comm);
    inline void postRecvs(const MpiComm& comm);
    template < std::invocable< Scalar&, Scalar > Combine >
    void wait(Combine&& combine);

    template < util::KokkosView_c View >
    void setOwned(View&& view)
        requires(std::remove_cvref_t< View >::rank() == 2 and
                 not std::is_const_v< typename std::remove_cvref_t< View >::value_type >)
    {
        setOwned(util::flatten(view), view.stride(0));
    }
    template < util::KokkosView_c View >
    void setShared(View&& view)
        requires(std::remove_cvref_t< View >::rank() == 2)
    {
        setShared(util::flatten(view), view.stride(0));
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
template < std::integral GlobalIndex >
using rank_ind_map_t = util::ArrayOwner< std::vector< GlobalIndex > >;

template < std::signed_integral GlobalIndex >
auto makeOwnedMap(const MpiComm&                 comm,
                  std::span< const GlobalIndex > owned_inds,
                  std::span< const GlobalIndex > shared_inds) -> rank_ind_map_t< GlobalIndex >
{
    const auto owned_set    = robin_hood::unordered_flat_set< GlobalIndex >(owned_inds.begin(), owned_inds.end());
    const auto is_owned_ind = [&owned_set](GlobalIndex i) {
        return owned_set.contains(i);
    };
    const int  my_rank          = comm.getRank();
    const auto comm_sz          = static_cast< size_t >(comm.getSize());
    auto       retval           = rank_ind_map_t< GlobalIndex >(comm_sz);
    const auto save_in_nbr_inds = [&](std::span< const GlobalIndex > shared, int rank) {
        if (rank != my_rank)
            std::ranges::copy_if(shared, std::back_inserter(retval.at(rank)), is_owned_ind);
    };
    util::staggeredAllGather(comm, shared_inds, save_in_nbr_inds);
    return retval;
}

template < std::signed_integral GlobalIndex >
auto makeSharedMap(const MpiComm& comm, const rank_ind_map_t< GlobalIndex >& owned_map) -> rank_ind_map_t< GlobalIndex >
{
    const auto comm_sz = static_cast< size_t >(comm.getSize());
    auto owned_nbr_sz  = util::ArrayOwner< int >{owned_map | std::views::transform(&std::vector< GlobalIndex >::size)};
    auto shared_nbr_sz = util::ArrayOwner< int >(comm_sz);
    comm.allToAllAsync(owned_nbr_sz, shared_nbr_sz).wait();
    auto retval = rank_ind_map_t< GlobalIndex >(comm_sz);
    auto reqs   = std::vector< MpiComm::Request >{};
    for (int rank = 0; auto& shared_inds : retval)
    {
        const auto num_shared_inds = shared_nbr_sz.at(static_cast< size_t >(rank));
        if (num_shared_inds != 0)
        {
            shared_inds.resize(num_shared_inds);
            reqs.push_back(comm.receiveAsync(shared_inds, rank, 0));
        }
        ++rank;
    }
    for (int rank = 0; const auto& owned_inds : owned_map)
    {
        if (owned_inds.size() != 0)
            reqs.push_back(comm.sendAsync(owned_inds, rank, 0));
        ++rank;
    }
    MpiComm::Request::waitAll(reqs);
    return retval;
}

template < std::signed_integral GlobalIndex >
bool checkSharedIndexing(const rank_ind_map_t< GlobalIndex >& shared_map, std::span< const GlobalIndex > shared_inds)
{
    const auto g2l         = util::IndexMap< GlobalIndex >{shared_inds};
    auto       shared_lids = shared_map | std::views::join | std::views::transform(g2l);
    const auto num_shared  = std::transform_reduce(
        shared_map.begin(), shared_map.end(), 0uz, std::plus{}, [](const auto& vec) { return vec.size(); });
    return std::ranges::equal(shared_lids, std::views::iota(0uz, num_shared));
}

template < std::signed_integral GlobalIndex >
auto getNeighborRanks(const rank_ind_map_t< GlobalIndex >& map) -> util::ArrayOwner< int >
{
    std::vector< int > retval;
    for (const auto& [i, vec] : map | std::views::enumerate)
        if (not vec.empty())
            retval.push_back(static_cast< int >(i));
    return {retval};
}

template < typename LocalIndex >
struct FlattenMapResult
{
    util::ArrayOwner< int >      nbr_ranks;
    util::CrsGraph< LocalIndex > inds;
};

template < std::signed_integral LocalIndex, std::signed_integral GlobalIndex >
auto flattenOwnedMap(const rank_ind_map_t< GlobalIndex >& owned_map,
                     std::span< const GlobalIndex >       owned_inds) -> FlattenMapResult< LocalIndex >
{
    constexpr auto not_empty          = util::negatePredicate(&std::vector< GlobalIndex >::empty);
    constexpr auto get_size           = &std::vector< GlobalIndex >::size;
    auto           owned_map_filtered = owned_map | std::views::filter(not_empty);
    auto           sizes              = owned_map_filtered | std::views::transform(get_size);
    auto           owned_map_flat     = util::CrsGraph< LocalIndex >(sizes);
    const auto     g2l                = util::IndexMap< GlobalIndex, LocalIndex >{owned_inds};
    std::ranges::transform(owned_map_filtered | std::views::join, owned_map_flat.data(), g2l);
    return {getNeighborRanks(owned_map), std::move(owned_map_flat)};
}

template < typename LocalIndex >
struct DeflateMapResult
{
    util::ArrayOwner< int >        nbr_ranks;
    util::ArrayOwner< LocalIndex > offsets;
};

template < std::signed_integral LocalIndex, std::signed_integral GlobalIndex >
auto deflateSharedMap(const rank_ind_map_t< GlobalIndex >& shared_map) -> DeflateMapResult< LocalIndex >
{
    auto           nbr_ranks = getNeighborRanks(shared_map);
    constexpr auto not_empty = util::negatePredicate(&std::vector< GlobalIndex >::empty);
    constexpr auto get_size  = &std::vector< GlobalIndex >::size;
    auto sizes      = shared_map | std::views::filter(not_empty) | std::views::transform(get_size) | std::views::common;
    auto offsets    = util::ArrayOwner< LocalIndex >(nbr_ranks.size() + 1);
    offsets.front() = 0;
    std::inclusive_scan(sizes.begin(), sizes.end(), std::next(offsets.begin()));
    return {std::move(nbr_ranks), std::move(offsets)};
}
} // namespace detail

template < std::signed_integral LocalIndex >
template < std::signed_integral GlobalIndex >
ImportExportContext< LocalIndex >::ImportExportContext(const MpiComm&                 comm,
                                                       std::span< const GlobalIndex > owned_inds,
                                                       std::span< const GlobalIndex > shared_inds)
{
    util::throwingAssert(std::ranges::is_sorted(owned_inds), "Owned GIDs must be sorted");
    const auto owned_map  = detail::makeOwnedMap(comm, owned_inds, shared_inds);
    const auto shared_map = detail::makeSharedMap(comm, owned_map);
    util::throwingAssert(detail::checkSharedIndexing(shared_map, shared_inds),
                         "Shared GIDs must be sorted by owning rank, and sub-sorted by GID");
    auto [owned_nbrs, owned_map_flat]      = detail::flattenOwnedMap< LocalIndex >(owned_map, owned_inds);
    auto [shared_nbrs, shared_ind_offsets] = detail::deflateSharedMap< LocalIndex >(shared_map);
    const auto nbrs                        = std::array{std::span{owned_nbrs}, std::span{shared_nbrs}};
    m_nbrs                                 = util::ArrayOwner< int >{nbrs | std::views::join};
    m_num_owned_nbrs                       = owned_nbrs.size();
    m_owned_inds                           = std::move(owned_map_flat);
    m_shared_ind_offsets                   = std::move(shared_ind_offsets);
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
bool ImportExportBase< Scalar, LocalIndex >::isOwnedSizeSufficient(size_t size) const
{
    return size >= (m_num_vecs - 1) * m_owned_stride + m_max_owned + 1;
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
bool ImportExportBase< Scalar, LocalIndex >::isSharedSizeSufficient(size_t size) const
{
    return size >= (m_num_vecs - 1) * m_shared_stride + m_context->getNumSharedInds();
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
void Import< Scalar, LocalIndex >::setOwned(std::span< const Scalar > owned_vals, size_t owned_stride)
{
    Base::m_owned_stride = owned_stride;
    util::throwingAssert(Base::isOwnedSizeSufficient(owned_vals.size()));
    m_owned = owned_vals.data();
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
void Import< Scalar, LocalIndex >::setShared(std::span< Scalar > shared_vals, size_t shared_stride)
{
    Base::m_shared_stride = shared_stride;
    util::throwingAssert(Base::isSharedSizeSufficient(shared_vals.size()));
    m_shared = shared_vals.data();
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
void Import< Scalar, LocalIndex >::postComms(const MpiComm& comm)
{
    util::throwingAssert(m_owned, "`setOwned` must be called before `postComms`");
    util::throwingAssert(m_shared, "`setShared` must be called before `postComms`");
    postRecvs(comm);
    postSends(comm);
    m_recv_complete.clear(std::memory_order_release);
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
void Import< Scalar, LocalIndex >::postRecvs(const MpiComm& comm)
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

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
void Import< Scalar, LocalIndex >::postSends(const MpiComm& comm)
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

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
bool Import< Scalar, LocalIndex >::tryReceive()
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

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
void Import< Scalar, LocalIndex >::packSends()
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
        util::tbb::parallelFor(std::views::iota(size_t{0}, owned_inds.size()), pack_ind);
    };
    util::tbb::parallelFor(std::views::iota(size_t{0}, Base::m_num_vecs), pack_vec);
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
auto Import< Scalar, LocalIndex >::getRecvRequests() -> std::span< MpiComm::Request >
{
    return {Base::m_requests | std::views::take(Base::m_context->getSharedNbrs().size() * Base::m_num_vecs)};
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
auto Import< Scalar, LocalIndex >::getSendRequests() -> std::span< MpiComm::Request >
{
    return {Base::m_requests | std::views::drop(Base::m_context->getSharedNbrs().size() * Base::m_num_vecs)};
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
void Export< Scalar, LocalIndex >::setOwned(std::span< Scalar > owned_vals, size_t owned_stride)
{
    Base::m_owned_stride = owned_stride;
    util::throwingAssert(Base::isOwnedSizeSufficient(owned_vals.size()));
    m_owned = owned_vals.data();
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
void Export< Scalar, LocalIndex >::setShared(std::span< const Scalar > shared_vals, size_t shared_stride)
{
    Base::m_shared_stride = shared_stride;
    util::throwingAssert(Base::isSharedSizeSufficient(shared_vals.size()));
    m_shared = shared_vals.data();
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
void Export< Scalar, LocalIndex >::postRecvs(const lstr::MpiComm& comm)
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

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
void Export< Scalar, LocalIndex >::postSends(const MpiComm& comm)
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

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
auto Export< Scalar, LocalIndex >::getRecvRequests() -> std::span< MpiComm::Request >
{
    return {Base::m_requests | std::views::take(Base::m_context->getOwnedNbrs().size() * Base::m_num_vecs)};
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
auto Export< Scalar, LocalIndex >::getSendRequests() -> std::span< MpiComm::Request >
{
    return {Base::m_requests | std::views::drop(Base::m_context->getOwnedNbrs().size() * Base::m_num_vecs)};
}

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
template < std::invocable< Scalar&, Scalar > Combine >
void Export< Scalar, LocalIndex >::unpack(Combine&& combine)
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

template < Arithmetic_c Scalar, std::signed_integral LocalIndex >
template < std::invocable< Scalar&, Scalar > Combine >
void Export< Scalar, LocalIndex >::wait(Combine&& combine)
{
    MpiComm::Request::waitAll(getRecvRequests());
    unpack(std::forward< Combine >(combine));
    MpiComm::Request::waitAll(getSendRequests());
}
} // namespace lstr::comm
#endif // L3STER_COMM_IMPORTEXPORT_HPP
