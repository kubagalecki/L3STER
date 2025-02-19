#ifndef L3STER_UTIL_SEGMENTEDOWNERSHIP_HPP
#define L3STER_UTIL_SEGMENTEDOWNERSHIP_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/util/ArrayOwner.hpp"

#include <map>

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

    // Utilities for computing neighbor info

    // Rank to indices
    using NeighborInfo = std::map< int, std::vector< T > >;

    // Distribution of ownership among the ranks of the communicator
    [[nodiscard]] auto getOwnershipDist(const MpiComm& comm) const -> util::ArrayOwner< T >
    {
        auto num_owned = std::views::single(static_cast< T >(owned().size()));
        auto retval    = util::ArrayOwner< T >(static_cast< size_t >(comm.getSize()));
        comm.allGather(num_owned, retval.begin());
        std::inclusive_scan(retval.begin(), retval.end(), retval.begin());
        return retval;
    }

    // Who owns my shared indices?
    [[nodiscard]] auto computeOutNbrInfo(std::span< const T > own_dist) const -> NeighborInfo
    {
        const auto get_owner = [&](T i) {
            return static_cast< int >(std::distance(own_dist.begin(), std::ranges::upper_bound(own_dist, i)));
        };
        auto retval = std::map< int, std::vector< T > >{};
        for (auto s : shared())
            retval[get_owner(s)].push_back(s);
        return retval;
    }

    // Whose shared indices do I own?
    [[nodiscard]] auto computeInNbrInfo(const MpiComm& comm, const NeighborInfo& out_nbr_info) const -> NeighborInfo
    {
        const auto               comm_sz = static_cast< size_t >(comm.getSize());
        util::ArrayOwner< char > out_nbr_bmp(comm_sz, false), in_nbr_bmp(comm_sz);
        for (const auto& [rank, _] : out_nbr_info)
            out_nbr_bmp.at(rank) = true;
        comm.allToAllAsync(out_nbr_bmp, in_nbr_bmp).wait();
        auto retval = NeighborInfo{};
        for (auto&& [rank, is_innbr] : in_nbr_bmp | std::views::enumerate)
            if (is_innbr)
                retval[static_cast< int >(rank)];
        auto reqs = std::vector< MpiComm::Request >{};
        reqs.reserve(retval.size() + out_nbr_info.size());
        for (const auto& [out_nbr, inds] : out_nbr_info)
            reqs.push_back(comm.sendAsync(inds, out_nbr, 0));
        auto posted_recv = util::DynamicBitset{retval.size()};
        while (posted_recv.count() != posted_recv.size())
            for (auto&& [i, info] : retval | std::views::enumerate)
            {
                if (posted_recv.test(i))
                    continue;
                auto& [in_nbr, inds]       = info;
                const auto [status, ready] = comm.probeAsync(in_nbr, 0);
                if (ready)
                {
                    inds.resize(static_cast< size_t >(status.template numElems< T >()));
                    reqs.push_back(comm.receiveAsync(inds, in_nbr, 0));
                    posted_recv.set(i);
                }
            }
        MpiComm::Request::waitAll(reqs);
        return retval;
    }

private:
    T                     m_owned_begin = 0, m_owned_end = 0;
    util::ArrayOwner< T > m_shared;
};
} // namespace lstr::util
#endif // L3STER_UTIL_SEGMENTEDOWNERSHIP_HPP
