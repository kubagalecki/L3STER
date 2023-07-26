#ifndef L3STER_ASSEMBLY_SPARSITYGRAPH_HPP
#define L3STER_ASSEMBLY_SPARSITYGRAPH_HPP

#include "l3ster/dofs/DofsFromNodes.hpp"
#include "l3ster/dofs/MakeTpetraMap.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/CrsGraph.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/IndexMap.hpp"
#include "l3ster/util/StaticVector.hpp"

namespace lstr::glob_asm
{
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
Teuchos::RCP< const tpetra_fecrsgraph_t >
makeSparsityGraph(const MpiComm&                                          comm,
                  const mesh::MeshPartition< orders... >&                 mesh,
                  const dofs::NodeToGlobalDofMap< problem_def.n_fields >& node_to_dof_map,
                  const dofs::NodeCondensationMap< CP >&                  cond_map,
                  util::ConstexprValue< problem_def >                     problemdef_ctwrapper)
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
    return retval;
}
} // namespace lstr::glob_asm
#endif // L3STER_ASSEMBLY_SPARSITYGRAPH_HPP
