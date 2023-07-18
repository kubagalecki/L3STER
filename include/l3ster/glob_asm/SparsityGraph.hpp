#ifndef L3STER_ASSEMBLY_SPARSITYGRAPH_HPP
#define L3STER_ASSEMBLY_SPARSITYGRAPH_HPP

#include "l3ster/dofs/MakeTpetraMap.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/CrsGraph.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/IndexMap.hpp"
#include "l3ster/util/StaticVector.hpp"

namespace lstr::detail
{
template < IndexRange_c auto dof_inds, size_t n_nodes, size_t dofs_per_node, CondensationPolicy CP >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&       nodes,
                      const NodeToGlobalDofMap< dofs_per_node >& node_dof_map,
                      const NodeCondensationMap< CP >&           cond_map,
                      util::ConstexprValue< dof_inds >           dofinds_ctwrpr = {})
{
    std::array< global_dof_t, std::ranges::size(dof_inds) * n_nodes > retval;
    std::ranges::copy(nodes | std::views::transform([&](n_id_t node) {
                          return util::getValuesAtInds(node_dof_map(cond_map.getCondensedId(node)), dofinds_ctwrpr);
                      }) | std::views::join,
                      begin(retval));
    return retval;
}

template < size_t n_nodes, size_t dofs_per_node >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&                              nodes,
                      const NodeToGlobalDofMap< dofs_per_node >&                        node_dof_map,
                      const NodeCondensationMap< CondensationPolicy::ElementBoundary >& cond_map)
{
    util::StaticVector< global_dof_t, dofs_per_node * n_nodes > retval;
    std::ranges::copy(
        nodes | std::views::transform([&](n_id_t node) { return node_dof_map(cond_map.getCondensedId(node)); }) |
            std::views::join |
            std::views::filter([](auto node) { return node != NodeToGlobalDofMap< dofs_per_node >::invalid_dof; }),
        std::back_inserter(retval));
    return retval;
}

template < IndexRange_c auto dof_inds, size_t n_nodes, size_t dofs_per_node, size_t num_maps, CondensationPolicy CP >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&                nodes,
                      const NodeToLocalDofMap< dofs_per_node, num_maps >& node_dof_map,
                      CondensationPolicyTag< CP >,
                      const util::ConstexprValue< dof_inds > dofinds_ctwrpr = {})
{
    using dof_array_t = std::array< local_dof_t, std::ranges::size(dof_inds) * n_nodes >;
    auto retval       = std::array< dof_array_t, num_maps >{};
    auto iters        = std::array< typename dof_array_t::iterator, num_maps >{};
    std::ranges::transform(retval, begin(iters), [](auto& arr) { return arr.begin(); });
    for (auto node : nodes)
    {
        const auto& all_dof_arrays = node_dof_map(node);
        for (size_t i = 0; const auto& all_dofs : all_dof_arrays)
        {
            iters[i] = util::copyValuesAtInds(all_dofs, iters[i], dofinds_ctwrpr);
            ++i;
        }
    }
    return retval;
}

template < size_t n_nodes, size_t dofs_per_node, size_t num_maps >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&                nodes,
                      const NodeToLocalDofMap< dofs_per_node, num_maps >& node_dof_map)
{
    using dof_vec_t = util::StaticVector< local_dof_t, dofs_per_node * n_nodes >;
    std::array< dof_vec_t, num_maps > retval;
    for (auto node : nodes)
    {
        const auto& all_dof_arrays = node_dof_map(node);
        for (size_t i = 0; const auto& all_dofs : all_dof_arrays)
        {
            std::ranges::copy_if(all_dofs, std::back_inserter(retval[i]), [](local_dof_t dof) {
                return dof != NodeToLocalDofMap< dofs_per_node, num_maps >::invalid_dof;
            });
            ++i;
        }
    }
    return retval;
}

template < IndexRange_c auto dof_inds, mesh::ElementType ET, el_o_t EO >
auto getSortedPrimaryDofs(const mesh::Element< ET, EO >&                         element,
                          const NodeToDofMap_c auto&                             node_dof_map,
                          const NodeCondensationMap< CondensationPolicy::None >& cond_map,
                          util::ConstexprValue< dof_inds >                       dofinds_ctwrpr = {})
{
    auto primary_nodes = getPrimaryNodesArray< CondensationPolicy::None >(element);
    std::ranges::sort(primary_nodes);
    return getDofsFromNodes(primary_nodes, node_dof_map, cond_map, dofinds_ctwrpr);
}

template < IndexRange_c auto  dof_inds,
           mesh::ElementType  ET,
           el_o_t             EO,
           size_t             max_dofs_per_node,
           CondensationPolicy CP >
auto getUnsortedPrimaryDofs(const mesh::Element< ET, EO >&                 element,
                            const NodeToGlobalDofMap< max_dofs_per_node >& node_dof_map,
                            const NodeCondensationMap< CP >&               cond_map,
                            util::ConstexprValue< dof_inds >               dofinds_ctwrpr = {})
{
    return getDofsFromNodes(getPrimaryNodesArray< CP >(element), node_dof_map, cond_map, dofinds_ctwrpr);
}

template < IndexRange_c auto  dof_inds,
           mesh::ElementType  ET,
           el_o_t             EO,
           size_t             max_dofs_per_node,
           size_t             num_maps,
           CondensationPolicy CP >
auto getUnsortedPrimaryDofs(const mesh::Element< ET, EO >&                          element,
                            const NodeToLocalDofMap< max_dofs_per_node, num_maps >& node_dof_map,
                            CondensationPolicyTag< CP >                             cond_policy,
                            util::ConstexprValue< dof_inds >                        dofinds_ctwrpr = {})
{
    return getDofsFromNodes(getPrimaryNodesArray< CP >(element), node_dof_map, cond_policy, dofinds_ctwrpr);
}

template < mesh::ElementType ET, el_o_t EO, size_t max_dofs_per_node, CondensationPolicy CP >
auto getUnsortedPrimaryDofs(const mesh::Element< ET, EO >&                 element,
                            const NodeToGlobalDofMap< max_dofs_per_node >& node_dof_map,
                            const NodeCondensationMap< CP >&               cond_map)
{
    return getDofsFromNodes(getPrimaryNodesArray< CP >(element), node_dof_map, cond_map);
}

template < mesh::ElementType ET, el_o_t EO, size_t max_dofs_per_node, size_t num_maps, CondensationPolicy CP >
auto getUnsortedPrimaryDofs(const mesh::Element< ET, EO >&                          element,
                            const NodeToLocalDofMap< max_dofs_per_node, num_maps >& node_dof_map,
                            CondensationPolicyTag< CP >)
{
    return getDofsFromNodes(getPrimaryNodesArray< CP >(element), node_dof_map);
}

struct NodeDofs
{
    std::vector< global_dof_t > dofs;
    size_t                      n_owned_dofs;
};

template < el_o_t... orders, size_t max_dofs_per_node, CondensationPolicy CP >
auto computeNodeDofs(const mesh::MeshPartition< orders... >&        mesh,
                     const NodeToGlobalDofMap< max_dofs_per_node >& node_to_dof_map,
                     const NodeCondensationMap< CP >&               cond_map) -> NodeDofs
{
    const auto get_node_dofs = [&node_to_dof_map](auto&& node_range) {
        return std::forward< decltype(node_range) >(node_range) | std::views::transform(node_to_dof_map) |
               std::views::join |
               std::views::filter([](auto dof) { return dof != NodeToGlobalDofMap< max_dofs_per_node >::invalid_dof; });
    };
    std::vector< global_dof_t > dofs;
    dofs.reserve(cond_map.getCondensedIds().size() * max_dofs_per_node);
    std::ranges::copy(get_node_dofs(getCondensedOwnedNodesView(mesh, cond_map)), std::back_inserter(dofs));
    const auto n_owned_dofs = dofs.size();
    std::ranges::copy(get_node_dofs(getCondensedGhostNodesView(mesh, cond_map)), std::back_inserter(dofs));
    dofs.shrink_to_fit();
    return {std::move(dofs), n_owned_dofs};
}

template < el_o_t... orders, auto problem_def, CondensationPolicy CP >
auto computeDofGraph(const mesh::MeshPartition< orders... >&                 mesh,
                     const NodeToGlobalDofMap< deduceNFields(problem_def) >& node_to_dof_map,
                     const NodeCondensationMap< CP >&                        cond_map,
                     std::span< const global_dof_t >                         owned_plus_shared_dofs,
                     util::ConstexprValue< problem_def >                     problem_def_ctwrapper)
    -> std::pair< util::CrsGraph< global_dof_t >, Kokkos::DualView< size_t*, tpetra_fecrsgraph_t::execution_space > >
{
    L3STER_PROFILE_FUNCTION;
    L3STER_PROFILE_REGION_BEGIN("Compute global to local DOF map");
    const auto global_to_local_dof_map = util::IndexMap{owned_plus_shared_dofs};
    L3STER_PROFILE_REGION_END("Compute global to local DOF map");

    const auto iterate_over_mesh = [&](auto&& element_kernel) {
        util::forEachConstexprParallel(
            [&]< auto dom_def >(util::ConstexprValue< dom_def >) {
                constexpr auto domain_id        = dom_def.first;
                constexpr auto covered_dof_inds = util::getTrueInds< dom_def.second >();
                mesh.visit(
                    [&](const auto& element) {
                        std::invoke(element_kernel, element, util::ConstexprValue< covered_dof_inds >{});
                    },
                    domain_id,
                    std::execution::par);
            },
            problem_def_ctwrapper);
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
    auto owned_map             = makeTpetraMap(owned_dofs, comm);
    auto owned_plus_shared_map = makeTpetraMap(owned_plus_shared_dofs, comm);
    return util::makeTeuchosRCP< tpetra_fecrsgraph_t >(
        std::move(owned_map), std::move(owned_plus_shared_map), std::move(row_sizes));
}

template < el_o_t... orders, detail::ProblemDef_c auto problem_def, CondensationPolicy CP >
Teuchos::RCP< const tpetra_fecrsgraph_t >
makeSparsityGraph(const MpiComm&                                          comm,
                  const mesh::MeshPartition< orders... >&                 mesh,
                  const NodeToGlobalDofMap< deduceNFields(problem_def) >& node_to_dof_map,
                  const NodeCondensationMap< CP >&                        cond_map,
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
} // namespace lstr::detail
#endif // L3STER_ASSEMBLY_SPARSITYGRAPH_HPP
