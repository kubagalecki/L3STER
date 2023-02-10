#ifndef L3STER_ASSEMBLY_SPARSITYGRAPH_HPP
#define L3STER_ASSEMBLY_SPARSITYGRAPH_HPP

#include "l3ster/assembly/MakeTpetraMap.hpp"
#include "l3ster/util/CrsGraph.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/IndexMap.hpp"

#include <bit>
#include <mutex>

namespace lstr::detail
{
template < IndexRange_c auto dof_inds, size_t n_nodes, size_t dofs_per_node >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&       nodes,
                      const NodeToGlobalDofMap< dofs_per_node >& map,
                      ConstexprValue< dof_inds >                 dofinds_ctwrpr = {})
{
    std::array< global_dof_t, std::ranges::size(dof_inds) * n_nodes > retval;
    std::ranges::copy(nodes | std::views::transform([&](n_id_t node) {
                          return getValuesAtInds(map(node), dofinds_ctwrpr);
                      }) | std::views::join,
                      begin(retval));
    return retval;
}

template < IndexRange_c auto dof_inds, size_t n_nodes, size_t dofs_per_node, size_t num_maps >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&                nodes,
                      const NodeToLocalDofMap< dofs_per_node, num_maps >& map,
                      ConstexprValue< dof_inds >                          dofinds_ctwrpr = {})
{
    using dof_array_t = std::array< local_dof_t, std::ranges::size(dof_inds) * n_nodes >;
    std::array< dof_array_t, num_maps >                    retval;
    std::array< typename dof_array_t::iterator, num_maps > iters;
    std::ranges::transform(retval, begin(iters), [](auto& arr) { return arr.begin(); });
    for (auto node : nodes)
    {
        const auto& all_dof_arrays = map(node);
        for (size_t i = 0; const auto& all_dofs : all_dof_arrays)
        {
            iters[i] = copyValuesAtInds(all_dofs, iters[i], dofinds_ctwrpr);
            ++i;
        }
    }
    return retval;
}

template < IndexRange_c auto dof_inds, ElementTypes T, el_o_t O >
auto getSortedElementDofs(const Element< T, O >&     element,
                          const NodeToDofMap_c auto& map,
                          ConstexprValue< dof_inds > dofinds_ctwrpr = {})
{
    auto nodes_copy = element.getNodes();
    std::ranges::sort(nodes_copy);
    return getDofsFromNodes(nodes_copy, map, dofinds_ctwrpr);
}

template < IndexRange_c auto dof_inds, ElementTypes T, el_o_t O >
auto getUnsortedElementDofs(const Element< T, O >&     element,
                            const NodeToDofMap_c auto& map,
                            ConstexprValue< dof_inds > dofinds_ctwrpr = {})
{
    return getDofsFromNodes(element.getNodes(), map, dofinds_ctwrpr);
}

struct NodeDofs
{
    std::vector< global_dof_t > dofs;
    size_t                      n_owned_dofs;
};

template < size_t max_dofs_per_node >
auto computeNodeDofs(const MeshPartition& mesh, const NodeToGlobalDofMap< max_dofs_per_node >& node_to_dof_map)
    -> NodeDofs
{
    const auto getNodeDofs = [&node_to_dof_map](auto node_span) {
        return node_span | std::views::transform(node_to_dof_map) | std::views::join |
               std::views::filter([](auto dof) { return dof != NodeToGlobalDofMap< max_dofs_per_node >::invalid_dof; });
    };
    std::vector< global_dof_t > dofs;
    dofs.reserve(mesh.getAllNodes().size() * max_dofs_per_node);
    std::ranges::copy(getNodeDofs(mesh.getOwnedNodes()), std::back_inserter(dofs));
    const auto n_owned_dofs = dofs.size();
    std::ranges::copy(getNodeDofs(mesh.getGhostNodes()), std::back_inserter(dofs));
    dofs.shrink_to_fit();
    return {std::move(dofs), n_owned_dofs};
}

inline auto makeContiguousGraph(std::vector< robin_hood::unordered_flat_set< n_id_t > > graph)
    -> util::CrsGraph< global_dof_t >
{
    auto retval =
        util::CrsGraph< global_dof_t >{graph | std::views::transform([](const auto& set) { return set.size(); })};
    util::tbb::parallelFor(std::views::iota(size_t{}, graph.size()), [&](size_t row_ind) {
        const auto row_span = retval(row_ind);
        std::ranges::copy(graph[row_ind], row_span.begin());
        std::ranges::sort(row_span);
    });
    return retval;
}

template < auto problem_def >
auto computeDofGraph(const MeshPartition&                                    mesh,
                     const NodeToGlobalDofMap< deduceNFields(problem_def) >& node_to_dof_map,
                     std::span< const global_dof_t >                         owned_plus_shared_dofs,
                     ConstexprValue< problem_def > problem_def_ctwrapper) -> util::CrsGraph< global_dof_t >
{
    const auto available_concurrency = static_cast< size_t >(oneapi::tbb::this_task_arena::max_concurrency());
    const auto num_mutexes           = std::bit_ceil(available_concurrency) << 6;
    auto       get_row_lock          = [mutexes = std::vector< std::mutex >(num_mutexes),
                         modulo_mask = static_cast< global_dof_t >(num_mutexes - 1)](global_dof_t dof_ind) mutable {
        const auto mutex_ind = dof_ind & modulo_mask;
        return std::unique_lock{mutexes[mutex_ind], std::defer_lock};
    };

    const auto global_to_local_dof_map = IndexMap{owned_plus_shared_dofs};
    auto       graph = std::vector< robin_hood::unordered_flat_set< n_id_t > >(owned_plus_shared_dofs.size());

    const auto process_domain = [&]< auto dom_def >(ConstexprValue< dom_def >) {
        constexpr auto  domain_id        = dom_def.first;
        constexpr auto& coverage         = dom_def.second;
        constexpr auto  covered_dof_inds = getTrueInds< coverage >();

        const auto process_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
            const auto element_dofs = getSortedElementDofs< covered_dof_inds >(element, node_to_dof_map);
            std::bitset< std::tuple_size_v< decltype(element_dofs) > > processed_nodes_mask;
            do
                for (size_t row_dof_ind = 0; auto row_dof : element_dofs)
                {
                    if (not processed_nodes_mask.test(row_dof_ind))
                    {
                        const auto local_dof = global_to_local_dof_map(row_dof);
                        auto       lock      = get_row_lock(local_dof);
                        if (lock.try_lock())
                        {
                            auto& node_entries = graph[local_dof];
                            node_entries.reserve(node_entries.size() + element_dofs.size());
                            for (auto col_dof : element_dofs)
                                node_entries.insert(col_dof);
                            lock.unlock();
                            processed_nodes_mask.set(row_dof_ind);
                        }
                    }
                    ++row_dof_ind;
                }
            while (not processed_nodes_mask.all());
        };
        mesh.visit(process_element, domain_id, std::execution::par);
    };
    forEachConstexprParallel(process_domain, problem_def_ctwrapper);
    return makeContiguousGraph(std::move(graph));
}

inline auto computeRowSizes(const util::CrsGraph< global_dof_t >& dof_graph) -> Kokkos::DualView< size_t* >
{
    Kokkos::DualView< size_t* > retval{"sparse graph row sizes", dof_graph.size()};
    auto                        host_view = retval.view_host();
    retval.modify_host();
    std::ranges::transform(std::views::iota(size_t{}, dof_graph.size()), host_view.data(), [&](auto row_dof) {
        return dof_graph(row_dof).size();
    });
    retval.sync_device();
    return retval;
}

inline auto makeDofMaps(std::span< const global_dof_t > owned,
                        std::span< const global_dof_t > owned_plus_shared,
                        const MpiComm&                  comm)
{
    return std::make_pair(makeTpetraMap(owned, comm), makeTpetraMap(owned_plus_shared, comm));
}

template < detail::ProblemDef_c auto problem_def >
Teuchos::RCP< const tpetra_fecrsgraph_t >
makeSparsityGraph(const MeshPartition&                                    mesh,
                  const NodeToGlobalDofMap< deduceNFields(problem_def) >& global_node_to_dof_map,
                  ConstexprValue< problem_def >                           problemdef_ctwrapper,
                  const MpiComm&                                          comm)
{
    const auto [all_dofs, n_owned_dofs] = computeNodeDofs(mesh, global_node_to_dof_map);
    const auto owned_plus_shared_dofs   = std::span{all_dofs};
    const auto owned_dofs               = owned_plus_shared_dofs.subspan(0, n_owned_dofs);
    const auto dof_graph = computeDofGraph(mesh, global_node_to_dof_map, owned_plus_shared_dofs, problemdef_ctwrapper);
    const auto row_sizes = computeRowSizes(dof_graph);
    const auto [owned_map, owned_plus_shared_map] = makeDofMaps(owned_dofs, owned_plus_shared_dofs, comm);
    auto retval = makeTeuchosRCP< tpetra_fecrsgraph_t >(owned_map, owned_plus_shared_map, row_sizes);
    retval->beginAssembly();
    for (size_t row_dof_ind = 0; auto row_dof : all_dofs)
        retval->insertGlobalIndices(row_dof, asTeuchosView(dof_graph(row_dof_ind++)));
    retval->endAssembly();
    return retval;
}
} // namespace lstr::detail
#endif // L3STER_ASSEMBLY_SPARSITYGRAPH_HPP
