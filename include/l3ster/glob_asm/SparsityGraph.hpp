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
using rank_to_their_ghost_nodes_map_t = robin_hood::unordered_flat_map< int, std::vector< n_id_t > >;
using node_to_owner_map_t             = robin_hood::unordered_flat_map< n_id_t, int >;

// Determine the mapping of ranks to their ghost nodes owned by the current partition
template < el_o_t... orders >
auto computeOwnedNeighborGhostNodes(const MpiComm& comm, const mesh::MeshPartition< orders... >& mesh)
    -> rank_to_their_ghost_nodes_map_t
{
    const auto                 max_n_ghosts = std::invoke([&] {
        const size_t my_n_ghosts = mesh.getGhostNodes().size();
        size_t       retval{};
        comm.allReduce(std::span{&my_n_ghosts, 1}, &retval, MPI_MAX);
        return retval;
    });
    util::ArrayOwner< n_id_t > send_buf(max_n_ghosts + 1), recv_buf(max_n_ghosts + 1), proc_buf(max_n_ghosts + 1);
    send_buf.front() = mesh.getGhostNodes().size();
    std::copy(mesh.getGhostNodes(), std::next(send_buf.begin()));

    auto       retval        = rank_to_their_ghost_nodes_map_t{};
    const auto process_recvd = [&](const util::ArrayOwner< n_id_t >& data_buf, int potential_nbr_rank) {
        const auto             rank_ghosts    = std::span{std::next(data_buf.begin()), data_buf.front()};
        std::vector< n_id_t >* nbr_ghosts_ptr = nullptr;
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

// Let your neighbors know that you are their neighbor
inline auto pingNeighbors(const MpiComm&                                                      comm,
                          const robin_hood::unordered_flat_map< int, std::vector< n_id_t > >& nbr_map)
    -> std::vector< int >
{
    const int                comm_size = comm.getSize();
    util::ArrayOwner< char > send_nbr_bmp(comm_size), recv_nbr_bmp(comm_size);
    std::ranges::fill(send_nbr_bmp, 0);
    for (int send_nbr_ind :
         nbr_map | std::views::transform([](const auto& rh_pair) -> const auto& { return rh_pair.first; }))
        send_nbr_bmp[send_nbr_ind] = 1;
    comm.allToAllAsync(send_nbr_bmp, recv_nbr_bmp).wait();
    auto retval = std::vector< int >{};
    for (int rank = 0; char is_my_nbr : recv_nbr_bmp)
    {
        if (is_my_nbr)
            retval.push_back(rank);
        ++rank;
    }
    return retval;
}

// Tell your neighbors which of their ghost nodes you own
inline auto communicateGhostNodesToNeighbors(const MpiComm&                         comm,
                                             const rank_to_their_ghost_nodes_map_t& nbr_map,
                                             const std::vector< int >&              recv_nbrs) -> node_to_owner_map_t
{
    auto requests = std::vector< MpiComm::Request >{};
    requests.reserve(nbr_map.size());

    for (const auto& [nbr, nbr_ghost_nodes] : nbr_map)
        requests.push_back(comm.sendAsync(nbr_ghost_nodes, nbr, 0));

    auto retval            = node_to_owner_map_t{};
    auto recvd_ghost_nodes = std::vector< n_id_t >{};
    for (int nbr : recv_nbrs)
    {
        const auto recv_status = comm.probe(nbr, 0);
        const int  recv_size   = MpiComm::countReceivedElems< n_id_t >(recv_status);
        recvd_ghost_nodes.resize(static_cast< size_t >(recv_size));
        comm.receive(recvd_ghost_nodes, nbr, 0);
        for (n_id_t ghost : recvd_ghost_nodes)
            retval.emplace(ghost, nbr);
    }

    MpiComm::Request::waitAll(requests);
    return retval;
}

struct NeighborInfo
{
    std::vector< int >  recv_neighbors;
    node_to_owner_map_t ghost_owners;
};

struct GhostGraphCols
{
    std::vector< int >          owners;
    std::vector< global_dof_t > dofs;
};

// For each neighbor, compute the global DOFs which I will be inserting into their column map, along with their owners
// Note that I may not be the owner of these DOFs, for example:
// - rank R1 owns DOF D1
// - rank R3 owns DOF D2
// - rank R2 owns element E1, which has DOFs D1 and D2 (both are ghost DOFs for R2)
// - rank R2 needs to tell R1 that it will be inserting into column D2, but that the owner of D2 is actually R3, so that
//   R1 can properly sort its column map
template < el_o_t... orders, ProblemDef problem_def, CondensationPolicy CP >
auto computeNeighborGhostDofs(int                                                     my_rank,
                              const mesh::MeshPartition< orders... >&                 mesh,
                              const dofs::NodeToGlobalDofMap< problem_def.n_fields >& node_to_global_dof_map,
                              const dofs::NodeCondensationMap< CP >&                  cond_map,
                              const NeighborInfo&                                     neighbor_info,
                              util::ConstexprValue< problem_def >                     probdef_ctwrpr)
    -> robin_hood::unordered_flat_map< int, GhostGraphCols >
{
    using dof_set_t                   = robin_hood::unordered_flat_set< global_dof_t >;
    using owner_to_dof_map_t          = robin_hood::unordered_flat_map< int, dof_set_t >;
    using rank_to_ghost_col_entries_t = robin_hood::unordered_flat_map< int, owner_to_dof_map_t >;
    auto rank_to_ghost_col_entries    = rank_to_ghost_col_entries_t{};

    const auto qualify_el_nodes = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        constexpr auto n_primary_nodes   = dofs::getNumPrimaryNodes< CP, ET, EO >();
        const auto     primary_nodes     = dofs::getPrimaryNodesArray< CP >(element);
        auto           retval            = std::array< util::StaticVector< n_id_t, n_primary_nodes >, 2 >{};
        auto& [owned_nodes, ghost_nodes] = retval;
        for (n_id_t node : primary_nodes)
            if (mesh.isGhostNode(node))
                ghost_nodes.push_back(node);
            else
                owned_nodes.push_back(node);
        return retval;
    };
    const auto get_global_dofs = [&]< size_t n_nodes, auto dom_def >(const util::StaticVector< n_id_t, n_nodes >& nodes,
                                                                     util::ConstexprValue< dom_def >) {
        using node_dof_vec_t     = util::StaticVector< global_dof_t, problem_def.n_fields >;
        auto       retval        = util::StaticVector< node_dof_vec_t, n_nodes >{};
        const auto get_node_dofs = [&](n_id_t node) {
            constexpr auto dofinds_ctwrpr = util::ConstexprValue< util::getTrueInds< dom_def.active_fields >() >{};
            return dofs::getNodeActiveDofs(node, node_to_global_dof_map, cond_map, dofinds_ctwrpr);
        };
        std::ranges::transform(nodes, std::back_inserter(retval), get_node_dofs);
        return retval;
    };
    const auto get_owners = [&]< size_t n_nodes >(const util::StaticVector< n_id_t, n_nodes >& nodes) {
        auto       retval         = util::StaticVector< int, n_nodes >{};
        const auto get_node_owner = [&](n_id_t node) {
            return neighbor_info.ghost_owners.at(node);
        };
        std::ranges::transform(nodes, std::back_inserter(retval), get_node_owner);
        return retval;
    };
    const auto process_domain = [&]< auto dom_def >(util::ConstexprValue< dom_def > dom_def_ctwrpr) {
        const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
            const auto [owned_nodes, ghost_nodes] = qualify_el_nodes(element);
            const auto ghost_node_owners          = get_owners(ghost_nodes);
            const auto owned_dofs                 = get_global_dofs(owned_nodes, dom_def_ctwrpr);
            const auto ghost_dofs                 = get_global_dofs(ghost_nodes, dom_def_ctwrpr);
            for (int ghost_owner : ghost_node_owners)
            {
                auto& target_map    = rank_to_ghost_col_entries[ghost_owner];
                auto& my_insert_set = target_map[my_rank];
                for (global_dof_t dof : owned_dofs | std::views::join)
                    my_insert_set.insert(dof);
                for (size_t i = 0; n_id_t node : ghost_nodes)
                {
                    const auto  node_owner = ghost_node_owners[i];
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
    util::forEachConstexprParallel(process_domain, probdef_ctwrpr);

    constexpr auto flatten_rank_map_el = [](const auto& rme) {
        constexpr auto flatten_map_of_sets = [](const owner_to_dof_map_t& mos) {
            constexpr auto get_set_size = [](const auto& map_elem) {
                return map_elem.second.size();
            };
            const size_t n_entries = std::transform_reduce(mos.begin(), mos.end(), size_t{}, std::plus{}, get_set_size);
            auto         retval    = GhostGraphCols{};
            auto& [flat_owners, flat_dofs] = retval;
            flat_owners.reserve(n_entries);
            flat_dofs.reserve(n_entries);
            for (const auto& [owner, dofs] : mos)
            {
                std::ranges::fill_n(std::back_inserter(flat_owners), dofs.size(), owner);
                std::ranges::copy(dofs, std::back_inserter(flat_dofs));
                std::sort(std::prev(flat_dofs.end(), dofs.size()), flat_dofs.end());
            }
            return retval;
        };
        return std::make_pair(rme.first, flatten_map_of_sets(rme.second));
    };
    auto flat_view = rank_to_ghost_col_entries | std::views::transform(flatten_rank_map_el) | std::views::common;
    return robin_hood::unordered_flat_map< int, GhostGraphCols >(flat_view.begin(), flat_view.end());
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

inline void sortGraphRowsRemoveDuplicates(util::CrsGraph< global_dof_t >& graph, std::span< size_t > row_sizes)
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
                     const util::IndexMap< global_dof_t, local_dof_t >                        glob_to_loc_dof_map,
                     const dofs::NodeCondensationMap< CP >&                                   cond_map,
                     util::ConstexprValue< problem_def >                                      probdef_ctwrpr,
                     const Kokkos::DualView< size_t*, tpetra_fecrsgraph_t::execution_space >& crs_row_sizes)
    -> util::CrsGraph< global_dof_t >
{
    L3STER_PROFILE_FUNCTION;
    const auto crs_row_sizes_host_view = crs_row_sizes.view_host();
    const auto row_sizes               = util::asSpan(crs_row_sizes_host_view);
    auto       retval                  = util::CrsGraph< global_dof_t >(row_sizes);
    std::ranges::fill(row_sizes, 0);

    const auto fill_domain = [&]< auto dom_def >(util::ConstexprValue< dom_def >) {
        const auto fill_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
            constexpr auto dof_inds        = util::getTrueInds< dom_def.active_fields >();
            constexpr auto dof_inds_ctwrpr = util::ConstexprValue< dof_inds >{};

            const auto element_dofs_global = std::invoke([&] {
                if constexpr (CP == CondensationPolicy::None)
                    return getUnsortedPrimaryDofs(element, node_to_global_dof_map, cond_map, dof_inds_ctwrpr);
                else if constexpr ((CP == CondensationPolicy::ElementBoundary))
                    return getUnsortedPrimaryDofs(element, node_to_global_dof_map, cond_map);
            });
            const auto nd                  = element_dofs_global.size();

            /*
            constexpr size_t max_el_dofs        = mesh::Element< ET, EO >::n_nodes * problem_def.n_fields;
            auto             element_dofs_local = util::StaticVector< local_dof_t, max_el_dofs >{};
            std::ranges::transform(element_dofs_global, std::back_inserter(element_dofs_local), glob_to_loc_dof_map);
             */

            for (global_dof_t global_dof : element_dofs_global)
            {
                const auto local_row = glob_to_loc_dof_map(global_dof);
                const auto row       = retval(local_row);
                const auto write_pos = std::atomic_ref{row_sizes[local_row]}.fetch_add(nd, std::memory_order_acq_rel);
                std::ranges::copy(element_dofs_global, std::next(row.begin(), write_pos));
            }
        };
        mesh.visit(fill_element, dom_def.domain, std::execution::par);
    };
    util::forEachConstexprParallel(fill_domain, probdef_ctwrpr);
    sortGraphRowsRemoveDuplicates(retval, row_sizes);
    return retval;
}

inline auto makeTpetraCrsGraph(const MpiComm&                  comm,
                               std::span< const global_dof_t > owned_dofs,
                               std::span< const global_dof_t > owned_plus_shared_dofs,
                               util::CrsGraph< global_dof_t >  local_graph,
                               const Kokkos::DualView< size_t*, tpetra_fecrsgraph_t::execution_space >& row_sizes)
{
    L3STER_PROFILE_FUNCTION;
    const auto owned_map             = dofs::makeTpetraMap(owned_dofs, comm);
    const auto owned_plus_shared_map = dofs::makeTpetraMap(owned_plus_shared_dofs, comm);
    auto       retval = util::makeTeuchosRCP< tpetra_fecrsgraph_t >(owned_map, owned_plus_shared_map, row_sizes);

    const auto row_sizes_host_view = row_sizes.view_host();
    const auto row_sizes_span      = util::asSpan(row_sizes_host_view);
    retval->beginAssembly();

    L3STER_PROFILE_REGION_BEGIN("Insert into Tpetra::FECrsGraph");
    for (size_t local_row = 0; size_t row_size : row_sizes_span)
    {
        const auto row_allocation = local_graph(local_row);
        const auto row_entries    = row_allocation.subspan(0, row_size);
        const auto global_row     = owned_plus_shared_dofs[local_row];
        retval->insertGlobalIndices(global_row, util::asTeuchosView(row_entries));
        ++local_row;
    }
    L3STER_PROFILE_REGION_END("Insert into Tpetra::FECrsGraph");

    local_graph = {}; // Explicitly deallocate to free memory for endAssembly()

#ifdef L3STER_PROFILE_EXECUTION
    comm.barrier();
#endif

    L3STER_PROFILE_REGION_BEGIN("Communicate data between ranks");
    auto out_str = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
    retval->describe(*out_str, Teuchos::VERB_EXTREME);

    retval->endAssembly();

    retval->describe(*out_str, Teuchos::VERB_EXTREME);
    L3STER_PROFILE_REGION_END("Communicate data between ranks");

#ifdef L3STER_PROFILE_EXECUTION
    comm.barrier();
#endif

    return retval;
}
} // namespace detail

template < el_o_t... orders, ProblemDef problem_def, CondensationPolicy CP >
auto makeSparsityGraph(const MpiComm&                                          comm,
                       const mesh::MeshPartition< orders... >&                 mesh,
                       const dofs::NodeToGlobalDofMap< problem_def.n_fields >& node_to_dof_map,
                       const dofs::NodeCondensationMap< CP >&                  cond_map,
                       util::ConstexprValue< problem_def >                     problemdef_ctwrapper)
    -> Teuchos::RCP< const tpetra_fecrsgraph_t >
{
    L3STER_PROFILE_FUNCTION;
    const auto [all_dofs, n_owned_dofs] = detail::computeNodeDofs(mesh, node_to_dof_map, cond_map);
    const auto global_to_local_dof_map  = util::IndexMap< global_dof_t, local_dof_t >{all_dofs};
    auto       row_sizes                = detail::computeMaxCrsGraphRowSizes(
        mesh, node_to_dof_map, global_to_local_dof_map, cond_map, problemdef_ctwrapper);
    auto crs_graph = detail::computeCrsGraph(
        mesh, node_to_dof_map, global_to_local_dof_map, cond_map, problemdef_ctwrapper, row_sizes);
    row_sizes.sync_device();
    const auto owned_plus_shared_dofs = std::span{all_dofs};
    const auto owned_dofs             = owned_plus_shared_dofs.subspan(0, n_owned_dofs);
    return detail::makeTpetraCrsGraph(comm, owned_dofs, owned_plus_shared_dofs, std::move(crs_graph), row_sizes);
}

/*
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
auto computeDofGraph(const mesh::MeshPartition< orders... >&                 mesh,
                     const dofs::NodeToGlobalDofMap< problem_def.n_fields >& node_to_dof_map,
                     const dofs::NodeCondensationMap< CP >&                  cond_map,
                     std::span< const global_dof_t >                         owned_plus_shared_dofs,
                     util::ConstexprValue< problem_def >                     probdef_ctwrpr)
    -> std::pair< util::CrsGraph< global_dof_t >, Kokkos::DualView< size_t*, tpetra_fecrsgraph_t::execution_space > >
{
    L3STER_PROFILE_FUNCTION;
    L3STER_PROFILE_REGION_BEGIN("Compute global to local DOF map");
    const auto global_to_local_dof_map = util::IndexMap{owned_plus_shared_dofs};
    L3STER_PROFILE_REGION_END("Compute global to local DOF map");

    constexpr auto n_fields          = problem_def.n_fields;
    const auto     iterate_over_mesh = [&](auto&& element_kernel) {
        util::forEachConstexprParallel(
            [&]< DomainDef< n_fields > dom_def >(util::ConstexprValue< dom_def >) {
                constexpr auto covered_dof_inds = util::getTrueInds< dom_def.active_fields >();
                mesh.visit(
                    [&](const auto& element) {
                        std::invoke(element_kernel, element, util::ConstexprValue< covered_dof_inds >{});
                    },
                    dom_def.domain,
                    std::execution::par);
            },
            probdef_ctwrpr);
    };

    auto crs_row_sizes_dual_view = Kokkos::DualView< size_t*, tpetra_fecrsgraph_t::execution_space >{
        "CRS graph row sizes", owned_plus_shared_dofs.size()};
    auto crs_row_sizes_host_view = crs_row_sizes_dual_view.view_host();
    crs_row_sizes_dual_view.modify_host();
    const auto crs_row_sizes = util::asSpan(crs_row_sizes_host_view);
    std::ranges::fill(crs_row_sizes, size_t{});

    const auto get_element_dofs = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element,
                                                                         auto dofinds_ctwrpr) {
        if constexpr (CP == CondensationPolicy::None)
            return getUnsortedPrimaryDofs(element, node_to_dof_map, cond_map, dofinds_ctwrpr);
        else if constexpr ((CP == CondensationPolicy::ElementBoundary))
            return getUnsortedPrimaryDofs(element, node_to_dof_map, cond_map);
    };

    const auto compute_max_row_sizes = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element,
                                                                              auto dofinds_ctwrpr) {
        const auto element_dofs = get_element_dofs(element, dofinds_ctwrpr);
        for (auto global_dof : element_dofs)
        {
            const auto local_dof = global_to_local_dof_map(global_dof);
            std::atomic_ref{crs_row_sizes[local_dof]}.fetch_add(element_dofs.size(), std::memory_order_relaxed);
        }
    };
    L3STER_PROFILE_REGION_BEGIN("Compute max CRS graph row entries");
    iterate_over_mesh(compute_max_row_sizes);
    L3STER_PROFILE_REGION_END("Compute max CRS graph row entries");

    auto graph = util::CrsGraph< global_dof_t >{crs_row_sizes};
    std::ranges::fill(crs_row_sizes, size_t{});

    const auto fill_graph = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element,
                                                                   auto                           dof_inds_ctwrapper) {
        const auto element_dofs = get_element_dofs(element, dof_inds_ctwrapper);
        for (auto global_dof : element_dofs)
        {
            const auto local_dof     = global_to_local_dof_map(global_dof);
            auto       local_row_pos = std::atomic_ref{crs_row_sizes[local_dof]};
            const auto write_offset  = local_row_pos.fetch_add(element_dofs.size(), std::memory_order_acq_rel);
            std::ranges::copy(element_dofs, std::next(graph(local_dof).begin(), write_offset));
        }
    };
    L3STER_PROFILE_REGION_BEGIN("Fill CRS graph");
    iterate_over_mesh(fill_graph);
    L3STER_PROFILE_REGION_END("Fill CRS graph");

    std::ranges::fill(crs_row_sizes, size_t{});

    const auto remove_duplicate_entries = [&](size_t local_dof) {
        const auto graph_row = graph(local_dof);
        std::ranges::sort(graph_row);
        const auto unique_end    = std::ranges::unique(graph_row).begin();
        crs_row_sizes[local_dof] = std::distance(graph_row.begin(), unique_end);
    };
    L3STER_PROFILE_REGION_BEGIN("Sort CRS graph rows and remove duplicates");
    util::tbb::parallelFor(std::views::iota(size_t{}, owned_plus_shared_dofs.size()), remove_duplicate_entries);
    L3STER_PROFILE_REGION_END("Sort CRS graph rows and remove duplicates");

    // Total number of local entries may not overflow local_dof_t (Tpetra limitation)
    const auto num_entries = std::reduce(std::execution::par_unseq, crs_row_sizes.begin(), crs_row_sizes.end());
    util::throwingAssert(num_entries <= static_cast< size_t >(std::numeric_limits< local_dof_t >::max()),
                         "Size of local adjacency graph exceeded allowed value. Consider using more MPI ranks.");

    crs_row_sizes_dual_view.sync_device();
    return std::make_pair(std::move(graph), std::move(crs_row_sizes_dual_view));
}

inline auto initCrsGraph(const MpiComm&                                                    comm,
                         std::span< const global_dof_t >                                   owned_dofs,
                         std::span< const global_dof_t >                                   owned_plus_shared_dofs,
                         Kokkos::DualView< size_t*, tpetra_fecrsgraph_t::execution_space > row_sizes)
{
    auto owned_map             = dofs::makeTpetraMap(owned_dofs, comm);
    auto owned_plus_shared_map = dofs::makeTpetraMap(owned_plus_shared_dofs, comm);
    return util::makeTeuchosRCP< tpetra_fecrsgraph_t >(
        std::move(owned_map), std::move(owned_plus_shared_map), std::move(row_sizes));
}

template < el_o_t... orders, ProblemDef problem_def, CondensationPolicy CP >
auto makeSparsityGraph(const MpiComm&                                          comm,
                       const mesh::MeshPartition< orders... >&                 mesh,
                       const dofs::NodeToGlobalDofMap< problem_def.n_fields >& node_to_dof_map,
                       const dofs::NodeCondensationMap< CP >&                  cond_map,
                       util::ConstexprValue< problem_def >                     problemdef_ctwrapper)
    -> Teuchos::RCP< const tpetra_fecrsgraph_t >
{
    L3STER_PROFILE_FUNCTION;
    const auto [all_dofs, n_owned_dofs] = computeNodeDofs(mesh, node_to_dof_map, cond_map);
    const auto owned_plus_shared_dofs   = std::span{all_dofs};
    const auto owned_dofs               = owned_plus_shared_dofs.subspan(0, n_owned_dofs);
    auto [dof_graph, row_sizes] =
        computeDofGraph(mesh, node_to_dof_map, cond_map, owned_plus_shared_dofs, problemdef_ctwrapper);
    auto retval = initCrsGraph(comm, owned_dofs, owned_plus_shared_dofs, row_sizes);
    retval->beginAssembly();
    L3STER_PROFILE_REGION_BEGIN("Insert into Tpetra::FECrsGraph");
    const auto row_sizes_host_view = row_sizes.view_host();
    for (size_t row_dof_ind = 0; auto row_dof : all_dofs)
    {
        const auto row_allocation = dof_graph(row_dof_ind);
        const auto row_entries    = row_allocation.subspan(0, row_sizes_host_view[row_dof_ind]);
        retval->insertGlobalIndices(row_dof, util::asTeuchosView(row_entries));
        ++row_dof_ind;
    }
    L3STER_PROFILE_REGION_END("Insert into Tpetra::FECrsGraph");
    dof_graph = {}; // explicitly deallocate to free memory for endAssembly()
    L3STER_PROFILE_REGION_BEGIN("Communicate data between ranks");
    retval->endAssembly();
    L3STER_PROFILE_REGION_END("Communicate data between ranks");
#ifdef L3STER_PROFILE_EXECUTION
    comm.barrier();
#endif
    return retval;
}
 */
} // namespace lstr::glob_asm
#endif // L3STER_GLOB_ASM_SPARSITYGRAPH_HPP
