#ifndef L3STER_GLOB_ASM_SPARSITYGRAPH_HPP
#define L3STER_GLOB_ASM_SPARSITYGRAPH_HPP

#include "l3ster/dofs/DofsFromNodes.hpp"
#include "l3ster/dofs/MakeTpetraMap.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/CrsGraph.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/IndexMap.hpp"
#include "l3ster/util/StaticVector.hpp"

namespace lstr::glob_asm
{
namespace detail
{
using rank_to_nodes_map_t        = robin_hood::unordered_flat_map< int, std::vector< n_id_t > >;
using node_to_owner_map_t        = robin_hood::unordered_flat_map< n_id_t, int >;
using dof_set_t                  = robin_hood::unordered_flat_set< global_dof_t >;
using rank_to_dofs_map_t         = robin_hood::unordered_flat_map< int, dof_set_t >;
using rank_to_rank_to_dofs_map_t = robin_hood::unordered_flat_map< int, rank_to_dofs_map_t >;

// Establish in-neighbors and which of their ghost nodes I own
template < el_o_t... orders >
auto computeInNeighborGhostNodes(const MpiComm& comm, const mesh::MeshPartition< orders... >& mesh)
    -> rank_to_nodes_map_t
{
    const auto                 max_n_ghosts = std::invoke([&] {
        const size_t my_n_ghosts = mesh.getGhostNodes().size();
        size_t       retval{};
        comm.allReduce(std::span{&my_n_ghosts, 1}, &retval, MPI_MAX);
        return retval;
    });
    util::ArrayOwner< n_id_t > send_buf(max_n_ghosts + 1), recv_buf(max_n_ghosts + 1), proc_buf(max_n_ghosts + 1);
    send_buf.front() = mesh.getGhostNodes().size();
    std::ranges::copy(mesh.getGhostNodes(), std::next(send_buf.begin()));

    auto       retval        = rank_to_nodes_map_t{};
    const auto process_recvd = [&](const util::ArrayOwner< n_id_t >& data_buf, int potential_nbr_rank) {
        const auto             rank_ghosts    = std::span{std::next(data_buf.begin()), data_buf.front()};
        std::vector< n_id_t >* nbr_ghosts_ptr = nullptr; // avoid repeated lookup
        for (n_id_t ghost : rank_ghosts)
            if (mesh.isOwnedNode(ghost))
            {
                if (not nbr_ghosts_ptr)
                    nbr_ghosts_ptr = std::addressof(retval[potential_nbr_rank]);
                nbr_ghosts_ptr->push_back(ghost);
            }
    };

    // Staggered all-gather pattern
    const int my_rank = comm.getRank(), comm_size = comm.getSize();
    auto      request = comm.broadcastAsync(my_rank == 0 ? send_buf : recv_buf, 0);
    for (int send_rank = 1; send_rank != comm_size; ++send_rank)
    {
        request.wait();
        std::swap(recv_buf, proc_buf);
        request = comm.broadcastAsync(my_rank == send_rank ? send_buf : recv_buf, send_rank);
        if (my_rank != send_rank - 1)
            process_recvd(proc_buf, send_rank - 1);
    }
    request.wait();
    if (my_rank != comm_size - 1)
        process_recvd(recv_buf, comm_size - 1);

    return retval;
}

inline auto computeInNeighbors(const rank_to_nodes_map_t& in_nbr_to_nodes) -> std::vector< int >
{
    return util::toVector(in_nbr_to_nodes | std::views::transform([](const auto& p) { return p.first; }));
}

// Notify in-neighbors that I am their out-neighbor
inline auto computeOutNeighbors(const MpiComm& comm, const rank_to_nodes_map_t& in_nbr_to_nodes) -> std::vector< int >
{
    const int                comm_size = comm.getSize();
    util::ArrayOwner< char > out_nbr_bmap(comm_size), in_nbr_bmap(comm_size);
    std::ranges::fill(out_nbr_bmap, 0);
    for (int out_nbr_rank : in_nbr_to_nodes | std::views::transform([](const auto& rh_pair) { return rh_pair.first; }))
        out_nbr_bmap[out_nbr_rank] = 1;
    comm.allToAllAsync(out_nbr_bmap, in_nbr_bmap).wait();
    auto retval = std::vector< int >{};
    retval.reserve(std::ranges::count(in_nbr_bmap, 1));
    for (int rank = 0; char is_in_nbr : in_nbr_bmap)
    {
        if (is_in_nbr)
            retval.push_back(rank);
        ++rank;
    }
    return retval;
}

// Tell my in-neighbors which of their ghost nodes I own
inline auto computeGhostNodeOwners(const MpiComm&             comm,
                                   const rank_to_nodes_map_t& in_nbr_to_nodes,
                                   const std::vector< int >&  out_nbrs) -> node_to_owner_map_t
{
    auto requests = std::vector< MpiComm::Request >{};
    requests.reserve(in_nbr_to_nodes.size());

    for (const auto& [nbr, nbr_ghost_nodes] : in_nbr_to_nodes)
        requests.push_back(comm.sendAsync(nbr_ghost_nodes, nbr, 0));

    auto retval            = node_to_owner_map_t{};
    auto recvd_ghost_nodes = std::vector< n_id_t >{};
    for (int nbr : out_nbrs)
    {
        const int recv_size = comm.probe(nbr, 0).numElems< n_id_t >();
        recvd_ghost_nodes.resize(static_cast< size_t >(recv_size));
        comm.receive(recvd_ghost_nodes, nbr, 0);
        for (n_id_t ghost : recvd_ghost_nodes)
            retval.emplace(ghost, nbr);
    }

    MpiComm::Request::waitAll(requests);
    return retval;
}

// For each out-neighbor, compute the global DOFs which I will be inserting into their column map, along with their
// owners, since I may not be the owner of these DOFs, for example:
// - rank R1 owns DOF D1
// - rank R3 owns DOF D2
// - rank R2 owns element E1, which has DOFs D1 and D2 (both are ghost DOFs for R2)
// - rank R2 needs to tell R1 that it will be inserting into column D2, but that the owner of D2 is actually R3, so that
//     R1 can properly sort its column map
template < el_o_t... orders, ProblemDef problem_def, CondensationPolicy CP >
auto computeOutNeighborDofInfo(int                                                     my_rank,
                               const mesh::MeshPartition< orders... >&                 mesh,
                               const dofs::NodeToGlobalDofMap< problem_def.n_fields >& node_to_global_dof_map,
                               const dofs::NodeCondensationMap< CP >&                  cond_map,
                               const node_to_owner_map_t&                              ghost_owners,
                               const std::vector< int >&                               out_nbrs,
                               util::ConstexprValue< problem_def > probdef_ctwrpr) -> rank_to_rank_to_dofs_map_t
{
    const auto qualify_el_nodes = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        constexpr auto n_primary_nodes   = dofs::getNumPrimaryNodes< CP, ET, EO >();
        auto           retval            = std::array< util::StaticVector< n_id_t, n_primary_nodes >, 2 >{};
        auto& [owned_nodes, ghost_nodes] = retval;
        for (n_id_t node : dofs::getPrimaryNodesView< CP >(element))
        {
            auto& target = mesh.isGhostNode(node) ? ghost_nodes : owned_nodes;
            target.push_back(node);
        }
        return retval;
    };
    const auto get_ghost_owners = [&]< size_t n_nodes >(const util::StaticVector< n_id_t, n_nodes >& nodes) {
        auto retval = util::StaticVector< int, n_nodes >{};
        std::ranges::transform(nodes, std::back_inserter(retval), [&](n_id_t n) { return ghost_owners.at(n); });
        return retval;
    };
    const auto get_global_dofs = [&]< size_t n_nodes, auto dom_def >(const util::StaticVector< n_id_t, n_nodes >& nodes,
                                                                     util::ConstexprValue< dom_def >) {
        using node_dof_vec_t     = util::StaticVector< global_dof_t, problem_def.n_fields >;
        auto       retval        = util::StaticVector< node_dof_vec_t, n_nodes >{};
        const auto get_node_dofs = [&](n_id_t node) {
            constexpr auto dofinds_ctwrpr = util::ConstexprValue< util::getTrueInds< dom_def.active_fields >() >{};
            return node_dof_vec_t{dofs::getNodeActiveDofs(node, node_to_global_dof_map, cond_map, dofinds_ctwrpr)};
        };
        std::ranges::transform(nodes, std::back_inserter(retval), get_node_dofs);
        return retval;
    };

    auto       retval         = rank_to_rank_to_dofs_map_t{};
    const auto process_domain = [&]< auto dom_def >(util::ConstexprValue< dom_def > dom_def_ctwrpr) {
        const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
            const auto [owned_nodes, ghost_nodes] = qualify_el_nodes(element);
            const auto ghost_node_owners          = get_ghost_owners(ghost_nodes);
            const auto owned_dofs                 = get_global_dofs(owned_nodes, dom_def_ctwrpr);
            const auto ghost_dofs                 = get_global_dofs(ghost_nodes, dom_def_ctwrpr);
            for (int ghost_owner : ghost_node_owners)
            {
                auto& target_map    = retval[ghost_owner];
                auto& my_insert_set = target_map[my_rank];
                for (global_dof_t dof : owned_dofs | std::views::join)
                    my_insert_set.insert(dof);
                for (size_t i = 0; int node_owner : ghost_node_owners)
                {
                    const auto& node_dofs  = ghost_dofs[i];
                    auto&       insert_set = target_map[node_owner];
                    for (global_dof_t dof : node_dofs)
                        insert_set.insert(dof);
                    ++i;
                }
            }
        };
        mesh.visit(process_element, dom_def.domain, std::execution::seq);
    };
    util::forConstexpr(process_domain, probdef_ctwrpr);

    // Deadlock avoidance: send empty message to out-nbrs without active DOFs
    for (int nbr : out_nbrs)
        retval[nbr];

    return retval;
}

// Tell my out-neighbors which columns I'll be writing to, and who owns those colums' GID rows
inline auto exchangeNeighborDofInfo(const MpiComm&                    comm,
                                    const std::vector< int >&         in_nbrs,
                                    const rank_to_rank_to_dofs_map_t& out_dof_info) -> rank_to_dofs_map_t
{
    struct FlatInfo
    {
        std::vector< int >          owners;
        std::vector< global_dof_t > dofs;
    };
    using rank_to_flat_rank_to_dofs_map_t = robin_hood::unordered_flat_map< int, FlatInfo >;
    constexpr auto flatten_neighbor_dof_map =
        [](const rank_to_rank_to_dofs_map_t& nbr_info) -> rank_to_flat_rank_to_dofs_map_t {
        static constexpr auto flatten_map_of_sets = [](const rank_to_dofs_map_t& mos) {
            constexpr auto get_set_size = [](const auto& map_elem) {
                return map_elem.second.size();
            };
            const size_t n_entries = std::transform_reduce(mos.begin(), mos.end(), size_t{}, std::plus{}, get_set_size);
            auto         retval    = FlatInfo{};
            auto& [flat_owners, flat_dofs] = retval;
            flat_owners.reserve(n_entries);
            flat_dofs.reserve(n_entries);
            for (const auto& [owner, dofs] : mos)
            {
                std::ranges::fill_n(std::back_inserter(flat_owners), dofs.size(), owner);
                std::ranges::copy(dofs, std::back_inserter(flat_dofs));
                std::ranges::sort(flat_dofs | std::views::reverse | std::views::take(dofs.size()), std::greater{});
            }
            return retval;
        };
        auto flat_view = nbr_info | std::views::transform([](const auto& pair) {
                             return std::make_pair(pair.first, flatten_map_of_sets(pair.second));
                         }) |
                         std::views::common;
        return {flat_view.begin(), flat_view.end()};
    };
    constexpr auto reconstitute_map_of_sets = [](const FlatInfo& mos_flat) -> rank_to_dofs_map_t {
        const auto& [owners, dofs] = mos_flat;
        auto retval                = rank_to_dofs_map_t{};
        auto dof_it                = dofs.begin();
        for (auto own_it = owners.begin(); own_it != owners.end();)
        {
            const auto cur_own  = *own_it;
            const auto cur_end  = std::find_if(own_it, owners.end(), [cur_own](int o) { return o != cur_own; });
            const auto cur_sz   = std::distance(own_it, cur_end);
            const auto dofs_end = std::next(dof_it, cur_sz);
            const auto cur_dofs = std::span{dof_it, dofs_end};
            auto&      cur_set  = retval[cur_own];
            for (global_dof_t dof : cur_dofs)
                cur_set.insert(dof);
            own_it = cur_end;
            dof_it = dofs_end;
        }
        return retval;
    };

    constexpr int owner_tag = 0, dof_tag = 1;
    auto          requests = std::vector< MpiComm::Request >{};

    const auto begin_send_info = [&](const rank_to_flat_rank_to_dofs_map_t& flat_map) {
        for (const auto& [target_rank, flat_info] : flat_map)
        {
            const auto& [owners, dofs] = flat_info;
            requests.push_back(comm.sendAsync(owners, target_rank, owner_tag));
            requests.push_back(comm.sendAsync(dofs, target_rank, dof_tag));
        }
    };
    const auto probe_sizes = [&]() -> std::vector< size_t > {
        const size_t sz     = in_nbrs.size();
        auto         retval = std::vector< size_t >(sz);
        for (auto probed_inds = util::DynamicBitset{sz}; probed_inds.count() != sz;)
            for (size_t i = 0; int nbr : in_nbrs)
            {
                if (not probed_inds.test(i))
                {
                    const auto [status, is_ready] = comm.probeAsync(nbr, owner_tag);
                    if (is_ready)
                    {
                        retval[i] = status.numElems< int >();
                        probed_inds.set(i);
                    }
                }
                ++i;
            }
        return retval;
    };
    const auto begin_recv_info = [&](const std::vector< size_t >& recv_sizes) -> FlatInfo {
        const auto total_sz  = std::reduce(recv_sizes.begin(), recv_sizes.end());
        auto       retval    = FlatInfo{};
        auto& [owners, dofs] = retval;
        owners.resize(total_sz);
        dofs.resize(total_sz);
        for (size_t i = 0, offset = 0; int nbr : in_nbrs)
        {
            const auto sz = recv_sizes[i++];
            requests.push_back(
                comm.receiveAsync(owners | std::views::drop(offset) | std::views::take(sz), nbr, owner_tag));
            requests.push_back(comm.receiveAsync(dofs | std::views::drop(offset) | std::views::take(sz), nbr, dof_tag));
            offset += sz;
        }
        return retval;
    };

    requests.reserve(2 * (in_nbrs.size() + out_dof_info.size()));
    const auto out_dof_info_flat = flatten_neighbor_dof_map(out_dof_info);
    begin_send_info(out_dof_info_flat);
    const auto recv_sizes       = probe_sizes();
    const auto in_nbr_info_flat = begin_recv_info(recv_sizes);
    MpiComm::Request::waitAll(requests);
    return reconstitute_map_of_sets(in_nbr_info_flat);
}

// Combine DOF info (DOF + owner) which I computed with that which was sent to me
inline auto combineDofInfo(rank_to_dofs_map_t dof_info, const rank_to_rank_to_dofs_map_t& out_dof_info)
    -> rank_to_dofs_map_t
{
    for (const auto& [_, info_map] : out_dof_info)
        for (const auto& [owner, dofs] : info_map)
        {
            auto& target_set = dof_info[owner];
            for (global_dof_t dof : dofs)
                target_set.insert(dof);
        }
    return dof_info;
}

// Default ordering used by Trilinos:
// first owned DOFs sorted by GID, then remaining DOFs sorted by owner rank, sub-sorted by GID
inline void
incorporateNeighborDofs(int my_rank, std::vector< global_dof_t >& owned_dofs, const rank_to_dofs_map_t& dof_info)
{
    if (const auto my_it = dof_info.find(my_rank); my_it != dof_info.end())
    {
        std::ranges::copy(my_it->second, std::back_inserter(owned_dofs));
        util::sortRemoveDup(owned_dofs);
    }

    auto ranks = util::toVector(dof_info | std::views::transform([](const auto& pair) { return pair.first; }));
    std::erase(ranks, my_rank);
    std::ranges::sort(ranks);
    for (const int owner : ranks)
    {
        const auto& dofs = dof_info.at(owner);
        std::ranges::copy(dofs, std::back_inserter(owned_dofs));
        std::ranges::sort(owned_dofs | std::views::drop(owned_dofs.size() - dofs.size()));
    }
    owned_dofs.shrink_to_fit();
}

template < el_o_t... orders, ProblemDef problem_def, CondensationPolicy CP >
auto computeColDofs(std::span< const global_dof_t >                         owned_dofs,
                    const MpiComm&                                          comm,
                    const mesh::MeshPartition< orders... >&                 mesh,
                    const dofs::NodeToGlobalDofMap< problem_def.n_fields >& node_to_dof_map,
                    const dofs::NodeCondensationMap< CP >&                  cond_map,
                    util::ConstexprValue< problem_def > probdef_ctwrpr) -> std::vector< global_dof_t >
{
    L3STER_PROFILE_FUNCTION;
    const auto my_rank               = comm.getRank();
    const auto in_nbr_to_my_node_map = computeInNeighborGhostNodes(comm, mesh);
    const auto in_nbrs               = computeInNeighbors(in_nbr_to_my_node_map);
    const auto out_nbrs              = computeOutNeighbors(comm, in_nbr_to_my_node_map);
    const auto node_to_owner_map     = computeGhostNodeOwners(comm, in_nbr_to_my_node_map, out_nbrs);
    const auto out_dof_info          = computeOutNeighborDofInfo(
        my_rank, mesh, node_to_dof_map, cond_map, node_to_owner_map, out_nbrs, probdef_ctwrpr);
    auto       in_dof_info = exchangeNeighborDofInfo(comm, in_nbrs, out_dof_info);
    const auto dof_info    = combineDofInfo(std::move(in_dof_info), out_dof_info);
    auto       retval      = std::vector< global_dof_t >{owned_dofs.begin(), owned_dofs.end()};
    incorporateNeighborDofs(my_rank, retval, dof_info);
    return retval;
}

struct NodeDofs
{
    std::vector< global_dof_t > dofs;
    size_t                      n_owned_dofs;
};

template < el_o_t... orders, size_t max_dofs_per_node, CondensationPolicy CP >
auto computeNodeDofs(const mesh::MeshPartition< orders... >&              mesh,
                     const dofs::NodeToGlobalDofMap< max_dofs_per_node >& node_to_dof_map,
                     const dofs::NodeCondensationMap< CP >&               cond_map) -> NodeDofs
{
    const auto get_node_dofs = [&node_to_dof_map](auto&& node_range) {
        return std::forward< decltype(node_range) >(node_range) | std::views::transform(node_to_dof_map) |
               std::views::join | std::views::filter([](auto dof) {
                   return dof != dofs::NodeToGlobalDofMap< max_dofs_per_node >::invalid_dof;
               });
    };
    std::vector< global_dof_t > dofs;
    dofs.reserve(cond_map.getCondensedIds().size() * max_dofs_per_node);
    std::ranges::copy(get_node_dofs(cond_map.getCondensedOwnedNodesView(mesh)), std::back_inserter(dofs));
    const auto n_owned_dofs = dofs.size();
    std::ranges::copy(get_node_dofs(cond_map.getCondensedGhostNodesView(mesh)), std::back_inserter(dofs));
    dofs.shrink_to_fit();
    return {std::move(dofs), n_owned_dofs};
}

template < el_o_t... orders, ProblemDef problem_def, CondensationPolicy CP >
auto computeMaxCrsGraphRowSizes(const mesh::MeshPartition< orders... >&                 mesh,
                                const dofs::NodeToGlobalDofMap< problem_def.n_fields >& node_to_global_dof_map,
                                const util::IndexMap< global_dof_t, local_dof_t >       global_to_local_dof_map,
                                const dofs::NodeCondensationMap< CP >&                  cond_map,
                                util::ConstexprValue< problem_def >                     probdef_ctwrpr)
    -> Kokkos::DualView< size_t*, tpetra_fecrsgraph_t::execution_space >
{
    L3STER_PROFILE_FUNCTION;
    using retval_t               = Kokkos::DualView< size_t*, tpetra_fecrsgraph_t::execution_space >;
    auto       retval            = retval_t{"CRS graph max row sizes", global_to_local_dof_map.size()};
    auto       retval_host_view  = retval.view_host();
    const auto crs_max_row_sizes = util::asSpan(retval_host_view);
    retval.modify_host();
    std::ranges::fill(crs_max_row_sizes, size_t{});
    const auto update_domain_counts = [&]< auto dom_def >(util::ConstexprValue< dom_def >) {
        const auto update_element_counts =
            [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
                constexpr auto dof_inds_ctwrpr = util::ConstexprValue< util::getTrueInds< dom_def.active_fields >() >{};
                const auto     element_dofs_global = std::invoke([&] {
                    if constexpr (CP == CondensationPolicy::None)
                        return getUnsortedPrimaryDofs(element, node_to_global_dof_map, cond_map, dof_inds_ctwrpr);
                    else if constexpr ((CP == CondensationPolicy::ElementBoundary))
                        return getUnsortedPrimaryDofs(element, node_to_global_dof_map, cond_map);
                });
                const auto     n_el_dofs           = element_dofs_global.size();
                for (local_dof_t local_dof : element_dofs_global | std::views::transform(global_to_local_dof_map))
                    std::atomic_ref{crs_max_row_sizes[local_dof]}.fetch_add(n_el_dofs, std::memory_order_relaxed);
            };
        mesh.visit(update_element_counts, dom_def.domain, std::execution::par);
    };
    util::forEachConstexprParallel(update_domain_counts, probdef_ctwrpr);
    return retval;
}

inline void sortGraphRowsRemoveDuplicates(util::CrsGraph< local_dof_t >& graph, std::span< size_t > row_sizes)
{
    L3STER_PROFILE_FUNCTION;
    const auto process_row = [&](size_t row_ind) {
        const auto row = graph(row_ind);
        std::ranges::sort(row);
        const auto erase_range = std::ranges::unique(row);
        const auto unique_size = row.size() - std::ranges::size(erase_range);
        row_sizes[row_ind]     = unique_size;
    };
    util::tbb::parallelFor(std::views::iota(0u, graph.getNRows()), process_row);
}

template < el_o_t... orders, ProblemDef problem_def, CondensationPolicy CP >
auto computeCrsGraph(const mesh::MeshPartition< orders... >&                                  mesh,
                     const dofs::NodeToGlobalDofMap< problem_def.n_fields >&                  node_to_global_dof_map,
                     const util::IndexMap< global_dof_t, local_dof_t >                        glob_to_loc_row_map,
                     const util::IndexMap< global_dof_t, local_dof_t >                        glob_to_loc_col_map,
                     const dofs::NodeCondensationMap< CP >&                                   cond_map,
                     util::ConstexprValue< problem_def >                                      probdef_ctwrpr,
                     const Kokkos::DualView< size_t*, tpetra_fecrsgraph_t::execution_space >& crs_row_sizes)
    -> util::CrsGraph< local_dof_t >
{
    L3STER_PROFILE_FUNCTION;
    const auto crs_row_sizes_host_view = crs_row_sizes.view_host();
    const auto row_sizes               = util::asSpan(crs_row_sizes_host_view);
    auto       retval                  = util::CrsGraph< local_dof_t >(row_sizes);
    std::ranges::fill(row_sizes, 0);

    const auto fill_domain = [&]< auto dom_def >(util::ConstexprValue< dom_def >) {
        const auto fill_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
            const auto element_dofs_global    = std::invoke([&] {
                if constexpr (CP == CondensationPolicy::None)
                {
                    constexpr auto dof_inds        = util::getTrueInds< dom_def.active_fields >();
                    constexpr auto dof_inds_ctwrpr = util::ConstexprValue< dof_inds >{};
                    return getUnsortedPrimaryDofs(element, node_to_global_dof_map, cond_map, dof_inds_ctwrpr);
                }
                else if constexpr ((CP == CondensationPolicy::ElementBoundary))
                    return getUnsortedPrimaryDofs(element, node_to_global_dof_map, cond_map);
            });
            const auto element_dofs_col_local = std::invoke([&] {
                constexpr size_t max_dofs      = mesh::Element< ET, EO >::n_nodes * problem_def.n_fields;
                auto             retval_nested = util::StaticVector< local_dof_t, max_dofs >{};
                std::ranges::transform(element_dofs_global, std::back_inserter(retval_nested), glob_to_loc_col_map);
                return retval_nested;
            });
            const auto n_dofs                 = element_dofs_global.size();
            const auto mem_order              = std::memory_order_acq_rel;
            for (local_dof_t local_row : element_dofs_global | std::views::transform(glob_to_loc_row_map))
            {
                const auto row       = retval(local_row);
                const auto write_pos = std::atomic_ref{row_sizes[local_row]}.fetch_add(n_dofs, mem_order);
                std::ranges::copy(element_dofs_col_local, std::next(row.begin(), write_pos));
            }
        };
        mesh.visit(fill_element, dom_def.domain, std::execution::par);
    };
    util::forEachConstexprParallel(fill_domain, probdef_ctwrpr);
    sortGraphRowsRemoveDuplicates(retval, row_sizes);
    return retval;
}

inline auto makeTpetraCrsGraph(const MpiComm&                  comm,
                               std::span< const global_dof_t > owned_row_dofs,
                               std::span< const global_dof_t > owned_plus_shared_row_dofs,
                               std::span< const global_dof_t > col_dofs,
                               util::CrsGraph< local_dof_t >   local_graph,
                               const Kokkos::DualView< size_t*, tpetra_fecrsgraph_t::execution_space >& row_sizes)
{
    L3STER_PROFILE_FUNCTION;
    const auto teuchos_comm          = util::makeTeuchosRCP< const Teuchos::MpiComm< int > >(comm.get());
    const auto owned_map             = dofs::makeTpetraMap(owned_row_dofs, teuchos_comm);
    const auto owned_plus_shared_map = dofs::makeTpetraMap(owned_plus_shared_row_dofs, teuchos_comm);
    const auto col_map               = dofs::makeTpetraMap(col_dofs, teuchos_comm);
    auto retval = util::makeTeuchosRCP< tpetra_fecrsgraph_t >(owned_map, owned_plus_shared_map, col_map, row_sizes);

    L3STER_PROFILE_REGION_BEGIN("Insert into Tpetra::FECrsGraph");
    retval->beginAssembly();
    const auto row_sizes_host_view = row_sizes.view_host();
    for (size_t local_row = 0; size_t row_size : util::asSpan(row_sizes_host_view))
    {
        const auto row_allocation = local_graph(local_row);
        const auto row_entries    = row_allocation.subspan(0, row_size);
        retval->insertLocalIndices(static_cast< int >(local_row), util::asTeuchosView(row_entries));
        ++local_row;
    }
    L3STER_PROFILE_REGION_END("Insert into Tpetra::FECrsGraph");

    local_graph = {}; // Explicitly deallocate to free memory for endAssembly()

#ifdef L3STER_PROFILE_EXECUTION
    comm.barrier();
#endif
    L3STER_PROFILE_REGION_BEGIN("Communicate data between ranks");
    retval->endAssembly();
#ifdef L3STER_PROFILE_EXECUTION
    comm.barrier();
#endif
    L3STER_PROFILE_REGION_END("Communicate data between ranks");

    return retval;
}
} // namespace detail

template < el_o_t... orders, ProblemDef problem_def, CondensationPolicy CP >
auto makeSparsityGraph(const MpiComm&                                          comm,
                       const mesh::MeshPartition< orders... >&                 mesh,
                       const dofs::NodeToGlobalDofMap< problem_def.n_fields >& node_to_dof_map,
                       const dofs::NodeCondensationMap< CP >&                  cond_map,
                       util::ConstexprValue< problem_def > probdef_ctwrpr) -> Teuchos::RCP< const tpetra_fecrsgraph_t >
{
    L3STER_PROFILE_FUNCTION;
    const auto [all_row_dofs, n_owned_dofs] = detail::computeNodeDofs(mesh, node_to_dof_map, cond_map);
    const auto owned_plus_shared_row_dofs   = std::span{all_row_dofs};
    const auto owned_row_dofs               = owned_plus_shared_row_dofs.subspan(0, n_owned_dofs);
    const auto col_dofs = detail::computeColDofs(owned_row_dofs, comm, mesh, node_to_dof_map, cond_map, probdef_ctwrpr);
    const auto global_to_local_row_dof_map = util::IndexMap< global_dof_t, local_dof_t >{all_row_dofs};
    const auto global_to_local_col_dof_map = util::IndexMap< global_dof_t, local_dof_t >{col_dofs};
    auto       row_sizes                   = detail::computeMaxCrsGraphRowSizes(
        mesh, node_to_dof_map, global_to_local_row_dof_map, cond_map, probdef_ctwrpr);
    auto crs_graph = detail::computeCrsGraph(mesh,
                                             node_to_dof_map,
                                             global_to_local_row_dof_map,
                                             global_to_local_col_dof_map,
                                             cond_map,
                                             probdef_ctwrpr,
                                             row_sizes);
    row_sizes.sync_device();
    return detail::makeTpetraCrsGraph(
        comm, owned_row_dofs, owned_plus_shared_row_dofs, col_dofs, std::move(crs_graph), row_sizes);
}
} // namespace lstr::glob_asm
#endif // L3STER_GLOB_ASM_SPARSITYGRAPH_HPP
