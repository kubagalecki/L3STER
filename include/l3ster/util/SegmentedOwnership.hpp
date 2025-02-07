#ifndef L3STER_UTIL_SEGMENTEDOWNERSHIP_HPP
#define L3STER_UTIL_SEGMENTEDOWNERSHIP_HPP

#include "l3ster/comm/ImportExport.hpp"
#include "l3ster/util/ArrayOwner.hpp"

namespace lstr::util
{
/// Represents segmented ownership: rank i owns elements [e_i, e_{i+1}), where e_0 = 0
template < std::integral T >
class SegmentedOwnership
{
public:
    SegmentedOwnership() = default;
    template < RangeOfConvertibleTo_c< T > Shared >
    SegmentedOwnership(T owned_begin, size_t num_owned, Shared&& shared)
        : m_owned_begin{owned_begin},
          m_owned_end{owned_begin + static_cast< T >(num_owned)},
          m_shared{std::forward< Shared >(shared)}
    {
        util::throwingAssert(std::ranges::none_of(m_shared, [&](T i) { return isOwned(i); }));
        std::ranges::sort(m_shared);
    }

    [[nodiscard]] auto owned() const { return std::views::iota(m_owned_begin, m_owned_end); }
    [[nodiscard]] auto shared() const -> std::span< const T > { return m_shared; }
    [[nodiscard]] bool isOwned(T i) const { return i >= m_owned_begin and i < m_owned_end; }
    [[nodiscard]] bool isShared(T i) const { return std::ranges::binary_search(m_shared, i); }
    [[nodiscard]] auto localSize() const -> size_t { return owned().size() + shared().size(); }
    [[nodiscard]] auto getLocalIndex(T gid) const -> size_t
    {
        return isOwned(gid)
                 ? static_cast< size_t >(gid - m_owned_begin)
                 : owned().size() +
                       static_cast< size_t >(std::distance(shared().begin(), std::ranges::lower_bound(shared(), gid)));
    }
    [[nodiscard]] auto getGlobalIndex(size_t lid) const -> T
    {
        return lid < owned().size() ? owned()[lid] : shared()[lid - owned().size()];
    }

    [[nodiscard]] auto makeCommContext(const MpiComm& comm) const
        -> std::shared_ptr< comm::ImportExportContext< local_dof_t > >
    {
        using context_t = comm::ImportExportContext< local_dof_t >;
        const auto o    = util::ArrayOwner< global_dof_t >{owned()};
        if constexpr (std::same_as< T, global_dof_t >)
            return std::make_shared< context_t >(comm, std::span{o}, std::span{m_shared});
        else
        {
            const auto s = util::ArrayOwner< global_dof_t >{shared()};
            return std::make_shared< context_t >(comm, std::span{o}, std::span{s});
        }
    }
    [[nodiscard]] auto getOwnershipDist(const MpiComm& comm) const -> util::ArrayOwner< T >
    {
        auto num_owned = std::views::single(static_cast< T >(owned().size()));
        auto retval    = util::ArrayOwner< T >(static_cast< size_t >(comm.getSize()));
        comm.allGather(num_owned, retval.begin());
        std::inclusive_scan(retval.begin(), retval.end(), retval.begin());
        return retval;
    }

private:
    T                     m_owned_begin = 0, m_owned_end = 0;
    util::ArrayOwner< T > m_shared;
};
} // namespace lstr::util
#endif // L3STER_UTIL_SEGMENTEDOWNERSHIP_HPP
