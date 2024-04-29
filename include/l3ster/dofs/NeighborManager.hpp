#ifndef L3STER_DOFS_NEIGHBORMANAGER_HPP
#define L3STER_DOFS_NEIGHBORMANAGER_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/dofs/NodeToDofMap.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/ArrayOwner.hpp"

namespace lstr::dofs
{
class NeighborManager
{
public:
    using nbr_map_t = std::map< int, util::ArrayOwner< global_dof_t > >;

    template < el_o_t... orders, size_t max_dofs_per_node, CondensationPolicy CP >
    NeighborManager(const mesh::MeshPartition< orders... >&        mesh,
                    const NodeToGlobalDofMap< max_dofs_per_node >& dof_map,
                    const NodeCondensationMap< CP >&               cond_map,
                    const MpiComm&                                 comm);

    const auto& getInNeighbors() const { return m_in; }
    const auto& getOutNeighbors() const { return m_out; }

private:
    nbr_map_t m_in, m_out;
};

namespace detail
{
template < el_o_t... orders, size_t max_dofs_per_node, CondensationPolicy CP >
auto getDofs(const mesh::MeshPartition< orders... >&        mesh,
             const NodeToGlobalDofMap< max_dofs_per_node >& dof_map,
             const NodeCondensationMap< CP >&               cond_map) -> std::array< std::vector< global_dof_t >, 2 >
{
    const auto get_dofs = [&](std::span< const n_id_t > nodes, std::vector< global_dof_t >& dofs) {
        dofs.reserve(max_dofs_per_node * nodes.size());
        std::ranges::copy(
            nodes | std::views::transform([&](n_id_t node) { return dof_map(cond_map.getCondensedId(node)); }) |
                std::views::join | std::views::filter([](auto node) {
                    return node != NodeToGlobalDofMap< max_dofs_per_node >::invalid_dof;
                }),
            std::back_inserter(dofs));
        util::sortRemoveDup(dofs);
    };
    auto retval           = std::array< std::vector< global_dof_t >, 2 >{};
    auto& [owned, shared] = retval;
    get_dofs(mesh.getOwnedNodes(), owned);
    get_dofs(mesh.getGhostNodes(), shared);
    return retval;
}

inline auto makeInMap(std::span< const global_dof_t > my_owned_dofs,
                      std::span< const global_dof_t > my_shared_dofs,
                      const MpiComm&                  comm) -> NeighborManager::nbr_map_t
{
    auto       retval             = NeighborManager::nbr_map_t{};
    const auto process_nbr_ghosts = [&](std::span< const global_dof_t > nbr_shared_dofs, int nbr_rank) {
        if (nbr_rank == comm.getRank())
            return;
        const auto is_mine = [&](auto dof) {
            return std::ranges::binary_search(my_owned_dofs, dof);
        };
        const auto num_mine = static_cast< size_t >(std::ranges::count_if(nbr_shared_dofs, is_mine));
        if (num_mine == 0)
            return;
        auto& dest = retval[nbr_rank];
        dest       = util::ArrayOwner< global_dof_t >(num_mine);
        std::ranges::copy_if(nbr_shared_dofs, dest.begin(), is_mine);
    };
    util::staggeredAllGather(comm, my_shared_dofs, process_nbr_ghosts);
    return retval;
}

inline auto makeOutMap(const NeighborManager::nbr_map_t& in_map, const MpiComm& comm) -> NeighborManager::nbr_map_t
{
    const auto                   comm_size = static_cast< size_t >(comm.getSize());
    util::ArrayOwner< unsigned > in_sz(comm_size), out_sz(comm_size);
    std::ranges::fill(in_sz, 0u);
    auto get_req = [reqs = std::vector< MpiComm::Request >{}]() mutable -> auto& {
        return reqs.emplace_back();
    };
    for (const auto& [rank, dofs] : in_map)
    {
        in_sz.at(rank) = static_cast< unsigned >(dofs.size());
        get_req()      = comm.sendAsync(dofs, rank, 0);
    }
    comm.allToAllAsync(in_sz, out_sz).wait();
    auto retval = NeighborManager::nbr_map_t{};
    for (int rank = 0; auto s : out_sz)
    {
        if (s)
        {
            auto& dofs = retval[rank];
            dofs       = util::ArrayOwner< global_dof_t >(s);
            get_req()  = comm.receiveAsync(dofs, rank, 0);
        }
        ++rank;
    }
    return retval;
}
} // namespace detail

template < el_o_t... orders, size_t max_dofs_per_node, CondensationPolicy CP >
NeighborManager::NeighborManager(const mesh::MeshPartition< orders... >&        mesh,
                                 const NodeToGlobalDofMap< max_dofs_per_node >& dof_map,
                                 const NodeCondensationMap< CP >&               cond_map,
                                 const MpiComm&                                 comm)
{
    const auto [my_owned_dofs, my_shared_dofs] = detail::getDofs(mesh, dof_map, cond_map);
    m_in                                       = detail::makeInMap(my_owned_dofs, my_shared_dofs, comm);
    m_out                                      = detail::makeOutMap(m_in, comm);
}
} // namespace lstr::dofs
#endif // L3STER_DOFS_NEIGHBORMANAGER_HPP
