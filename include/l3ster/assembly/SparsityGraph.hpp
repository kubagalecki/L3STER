#ifndef L3STER_ASSEMBLY_SPARSITYGRAPH_HPP
#define L3STER_ASSEMBLY_SPARSITYGRAPH_HPP

#include "l3ster/assembly/MakeTpetraMap.hpp"
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

class CrsEntries
{
    static constexpr size_t buf_size        = 1ul << 28;
    static constexpr size_t overflowed_size = std::numeric_limits< size_t >::max();

    [[nodiscard]] global_dof_t* getBuf(size_t row) noexcept { return std::next(m_buf.get(), row * m_row_buf_size); }
    [[nodiscard]] const global_dof_t* getBuf(size_t row) const noexcept
    {
        return std::next(m_buf.get(), row * m_row_buf_size);
    }

public:
    CrsEntries(size_t n_rows_)
        : m_buf{std::make_unique_for_overwrite< global_dof_t[] >(buf_size)},
          m_buf_sizes{std::make_unique< size_t[] >(n_rows_)},
          m_overflowed{std::make_unique< std::vector< global_dof_t >[] >(n_rows_)},
          m_row_buf_size{buf_size / (n_rows_ == 0 ? size_t{1} : n_rows_)},
          m_n_rows{n_rows_}
    {}

    [[nodiscard]] std::span< const global_dof_t > getRowEntries(size_t row) const noexcept
    {
        const auto row_size      = m_buf_sizes[row];
        const auto row_buf_start = getBuf(row);
        if (row_size != overflowed_size) [[likely]]
            return {row_buf_start, row_size};
        else [[unlikely]]
            return {m_overflowed[row].data(), m_overflowed[row].size()};
    }
    [[nodiscard]] std::span< global_dof_t > getRowEntriesForOverwrite(size_t row, size_t requested_size)
    {
        if (requested_size <= m_row_buf_size) [[likely]]
        {
            m_buf_sizes[row] = requested_size;
            return {getBuf(row), requested_size};
        }
        else [[unlikely]]
        {
            m_buf_sizes[row] = overflowed_size;
            m_overflowed[row].resize(requested_size);
            return {m_overflowed[row].data(), requested_size};
        }
    }
    [[nodiscard]] size_t size() const noexcept { return m_n_rows; }

private:
    std::unique_ptr< global_dof_t[] >                m_buf;
    std::unique_ptr< size_t[] >                      m_buf_sizes;
    std::unique_ptr< std::vector< global_dof_t >[] > m_overflowed;
    const size_t                                     m_row_buf_size;
    const size_t                                     m_n_rows;
};

template < auto problem_def >
auto calculateCrsData(const MeshPartition&                                        mesh,
                      ConstexprValue< problem_def >                               problem_def_ctwrapper,
                      const node_interval_vector_t< deduceNFields(problem_def) >& dof_intervals,
                      const std::vector< global_dof_t >&                          owned_plus_shared_dofs)
{
    using scratchpad_t = std::vector< global_dof_t >;
    std::mutex                   scratchpad_registry_mutex;
    std::vector< scratchpad_t* > scratchpad_registry;
    scratchpad_registry.reserve(std::thread::hardware_concurrency());
    const auto register_scratchpad = [&](auto& scratchpad) {
        std::scoped_lock lock{scratchpad_registry_mutex};
        scratchpad_registry.push_back(std::addressof(scratchpad));
    };
    const auto dealloc_scratchpads = [&] {
        for (auto scr_ptr : scratchpad_registry)
            *scr_ptr = scratchpad_t();
    };

    CrsEntries row_entries(owned_plus_shared_dofs.size());
    const auto merge_new_dofs = [&]< size_t n_new >(size_t row, const std::array< global_dof_t, n_new >& new_dofs) {
        thread_local scratchpad_t scratchpad;
        if (scratchpad.empty())
            register_scratchpad(scratchpad);

        const auto old_dofs = row_entries.getRowEntries(row);
        scratchpad.resize(old_dofs.size() + n_new);
        const auto union_end            = std::ranges::set_union(old_dofs, new_dofs, begin(scratchpad)).out;
        const auto union_range          = std::ranges::subrange(begin(scratchpad), union_end);
        const auto new_dofs_write_range = row_entries.getRowEntriesForOverwrite(row, union_range.size());
        std::memcpy(new_dofs_write_range.data(), union_range.data(), new_dofs_write_range.size_bytes());
    };

    const auto node_to_dof_map         = NodeToGlobalDofMap{mesh, dof_intervals};
    const auto global_to_local_dof_map = IndexMap< global_dof_t, std::uint32_t >{owned_plus_shared_dofs};

    const auto     available_concurrency = static_cast< unsigned >(oneapi::tbb::this_task_arena::max_concurrency());
    const unsigned num_mutexes           = std::bit_ceil(available_concurrency) << 6;
    const size_t   mutex_mod_mask        = num_mutexes - 1;
    std::vector< std::mutex > mutexes(num_mutexes);

    const auto process_domain = [&]< auto dom_def >(ConstexprValue< dom_def >) {
        constexpr auto  domain_id        = dom_def.first;
        constexpr auto& coverage         = dom_def.second;
        constexpr auto  covered_dof_inds = getTrueInds< coverage >();

        const auto process_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
            const auto element_dofs = getSortedElementDofs< covered_dof_inds >(element, node_to_dof_map);
            std::bitset< std::tuple_size_v< decltype(element_dofs) > > processed_rows;
            do
                for (size_t row_ind = 0; row_ind < element_dofs.size(); ++row_ind)
                {
                    if (processed_rows.test(row_ind))
                        continue;

                    const auto       local_row = global_to_local_dof_map(element_dofs[row_ind]);
                    std::unique_lock lock{mutexes[local_row & mutex_mod_mask], std::defer_lock};
                    if (lock.try_lock())
                    {
                        merge_new_dofs(local_row, element_dofs);
                        lock.unlock();
                        processed_rows.set(row_ind);
                    }
                }
            while (not processed_rows.all());
        };
        mesh.visit(process_element, domain_id, std::execution::par);
    };
    forEachConstexprParallel(process_domain, problem_def_ctwrapper);
    dealloc_scratchpads();
    return row_entries;
}

inline Kokkos::DualView< size_t* > getRowSizes(const CrsEntries& entries)
{
    Kokkos::DualView< size_t* > retval{"sparse graph row sizes", entries.size()};
    auto                        host_view = retval.view_host();
    retval.modify_host();
    util::tbb::parallelTransform(std::views::iota(0u, entries.size()), host_view.data(), [&entries](size_t row) {
        return entries.getRowEntries(row).size();
    });
    retval.sync_device();
    return retval;
}

template < detail::ProblemDef_c auto problem_def >
Teuchos::RCP< const tpetra_fecrsgraph_t >
makeSparsityGraph(const MeshPartition&                                        mesh,
                  ConstexprValue< problem_def >                               problemdef_ctwrapper,
                  const node_interval_vector_t< deduceNFields(problem_def) >& dof_intervals,
                  const MpiComm&                                              comm)
{
    auto owned_dofs = detail::getNodeDofs(mesh.getOwnedNodes(), dof_intervals);
    auto owned_map  = makeTpetraMap(owned_dofs, comm);
    auto owned_plus_shared_dofs =
        concatVectors(std::move(owned_dofs), detail::getNodeDofs(mesh.getGhostNodes(), dof_intervals));
    auto owned_plus_shared_map = makeTpetraMap(owned_plus_shared_dofs, comm);

    const auto row_entries = computeCrsNodeGraph(mesh, problemdef_ctwrapper);
    const auto row_sizes   = getRowSizes(row_entries);

    auto retval = makeTeuchosRCP< tpetra_fecrsgraph_t >(owned_map, owned_plus_shared_map, row_sizes);
    retval->beginAssembly();
    for (ptrdiff_t local_row = 0; local_row < static_cast< ptrdiff_t >(owned_plus_shared_dofs.size()); ++local_row)
    {
        const auto row_inds = row_entries.getRowEntries(local_row);
        retval->insertGlobalIndices(
            owned_plus_shared_dofs[local_row],
            Teuchos::ArrayView{row_inds.data(),
                               static_cast< Teuchos::ArrayView< size_t >::size_type >(row_inds.size())});
    }
    retval->endAssembly();
    return retval;
}

template < auto problem_def >
auto computeCrsNodeGraph(const MeshPartition& mesh, ConstexprValue< problem_def > problem_def_ctwrapper)
    -> std::vector< robin_hood::unordered_flat_set< n_id_t > >
{
    const auto available_concurrency = static_cast< size_t >(oneapi::tbb::this_task_arena::max_concurrency());
    const auto num_mutexes           = std::bit_ceil(available_concurrency) << 6;

    auto get_row_lock = [mutexes     = std::vector< std::mutex >(num_mutexes),
                         modulo_mask = num_mutexes - 1u](size_t local_node_ind) mutable {
        const auto mutex_ind = local_node_ind & modulo_mask;
        return std::unique_lock{mutexes[mutex_ind], std::defer_lock};
    };

    const auto global_to_local_node_map = IndexMap{mesh.getAllNodes()};
    auto       retval         = std::vector< robin_hood::unordered_flat_set< n_id_t > >(mesh.getAllNodes().size());
    const auto process_domain = [&]< auto dom_def >(ConstexprValue< dom_def >) {
        constexpr auto  domain_id        = dom_def.first;
        constexpr auto& coverage         = dom_def.second;
        constexpr auto  covered_dof_inds = getTrueInds< coverage >();

        const auto process_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
            const auto&                                              element_nodes = element.getNodes();
            std::bitset< ElementTraits< Element< T, O > >::n_nodes > processed_nodes_mask;
            do
                for (size_t node_ind = 0; auto row_node : element_nodes)
                {
                    if (not processed_nodes_mask.test(node_ind))
                    {
                        const auto row_node_local_ind = global_to_local_node_map(row_node);
                        auto       lock               = get_row_lock(row_node_local_ind);
                        if (lock.try_lock())
                        {
                            auto& node_entries = retval[row_node_local_ind];
                            node_entries.reserve(node_entries.size() + element_nodes.size());
                            for (auto col_node : element_nodes)
                                node_entries.insert(col_node);
                            lock.unlock();
                            processed_nodes_mask.set(node_ind);
                        }
                    }
                    ++node_ind;
                }
            while (not processed_nodes_mask.all());
        };
        mesh.visit(process_element, domain_id, std::execution::par);
    };
    forEachConstexprParallel(process_domain, problem_def_ctwrapper);
    return retval;
}

struct ContiguousCrsNodeGraphData
{
    std::unique_ptr< global_dof_t[] > entries;
    std::vector< size_t >             node_offsets;
};

template < size_t max_dofs_per_node >
auto initDofColumnEntries(const NodeToGlobalDofMap< max_dofs_per_node >&                 node_to_dof_map,
                          const std::vector< robin_hood::unordered_flat_set< n_id_t > >& node_graph)
    -> ContiguousCrsNodeGraphData
{
    const auto compute_row_size = [&](const robin_hood::unordered_flat_set< n_id_t >& row_nodes) {
        const auto size_range = row_nodes | std::views::transform(node_to_dof_map) |
                                std::views::transform([](const auto& node_dofs) {
                                    return std::ranges::count_if(node_dofs, [](global_dof_t dof) {
                                        return dof != NodeToGlobalDofMap< max_dofs_per_node >::invalid_dof;
                                    });
                                }) |
                                std::views::common;
        return std::reduce(std::ranges::cbegin(size_range), std::ranges::cend(size_range));
    };
    std::vector< size_t > node_offsets(node_graph.size() + 1);
    std::transform_inclusive_scan(std::execution::par_unseq,
                                  begin(node_graph),
                                  end(node_graph),
                                  std::next(begin(node_offsets)),
                                  std::plus{},
                                  compute_row_size,
                                  size_t{});
    auto entries = std::make_unique_for_overwrite< global_dof_t[] >(node_offsets.back());
    return {std::move(entries), std::move(node_offsets)};
}

template < detail::ProblemDef_c auto problem_def >
auto computeDofColumnEntriesForNodes(const MeshPartition&                                    mesh,
                                     const NodeToGlobalDofMap< deduceNFields(problem_def) >& node_to_dof_map,
                                     ConstexprValue< problem_def > problemdef_ctwrapper) -> ContiguousCrsNodeGraphData
{
    const auto node_graph         = computeCrsNodeGraph(mesh, problemdef_ctwrapper);
    auto       retval             = initDofColumnEntries(node_to_dof_map, node_graph);
    auto& [entries, node_offsets] = retval;
    util::tbb::parallelFor(std::views::iota(size_t{}, node_graph.size()), [&](size_t row_ind) {
        const auto row_span = std::span{std::next(entries.get(), node_offsets[row_ind]),
                                        std::next(entries.get(), node_offsets[row_ind + 1])};
        std::ranges::copy(node_graph[row_ind] | std::views::transform(node_to_dof_map) | std::views::join |
                              std::views::filter([](global_dof_t dof) {
                                  return dof != NodeToGlobalDofMap< deduceNFields(problem_def) >::invalid_dof;
                              }),
                          row_span.begin());
        std::ranges::sort(row_span);
    });
    return retval;
}

template < detail::ProblemDef_c auto problem_def >
auto makeDofMaps(const MeshPartition&                                        mesh,
                 const node_interval_vector_t< deduceNFields(problem_def) >& dof_intervals,
                 const MpiComm&                                              comm)
{
    constexpr auto n_fields = deduceNFields(problem_def);
    auto           dofs     = std::vector< global_dof_t >{};
    dofs.reserve(n_fields * mesh.getAllNodes().size());
    writeNodeDofs(mesh.getOwnedNodes(), dof_intervals, std::back_inserter(dofs));
    auto owned_map = makeTpetraMap(dofs, comm);
    writeNodeDofs(mesh.getGhostNodes(), dof_intervals, std::back_inserter(dofs));
    auto owned_plus_shared_map = makeTpetraMap(dofs, comm);
    return std::make_pair(std::move(owned_map), std::move(owned_plus_shared_map));
}

inline auto getRowSizes(const std::vector< size_t >& row_offsets) -> Kokkos::DualView< size_t* >
{
    Kokkos::DualView< size_t* > retval{"sparse graph row sizes", row_offsets.size() - 1};
    auto                        host_view = retval.view_host();
    retval.modify_host();
    std::ranges::transform(row_offsets | std::views::drop(1),
                           row_offsets | std::views::take(row_offsets.size() - 1),
                           host_view.data(),
                           std::minus{});
    retval.sync_device();
    return retval;
}

template < detail::ProblemDef_c auto problem_def >
Teuchos::RCP< const tpetra_fecrsgraph_t >
makeSparsityGraph(const MeshPartition&                                        mesh,
                  const node_interval_vector_t< deduceNFields(problem_def) >& dof_intervals,
                  const NodeToGlobalDofMap< deduceNFields(problem_def) >&     node_to_dof_map,
                  ConstexprValue< problem_def >                               problemdef_ctwrapper,
                  const MpiComm&                                              comm)
{
    // All rows corresponding to the DOFs of a specific node have the same column entries. We can reduce the memory
    // footprint by only storing these entries once
    const auto [entries, node_offsets] = computeDofColumnEntriesForNodes(mesh, dof_intervals, problemdef_ctwrapper);
    const auto row_sizes               = getRowSizes(node_offsets);
    auto [owned_map, owned_shared_map] = makeDofMaps(mesh, dof_intervals, comm);
    auto retval = makeTeuchosRCP< tpetra_fecrsgraph_t >(std::move(owned_map), std::move(owned_shared_map), row_sizes);
    retval->beginAssembly();
    for (size_t local_node_ind = 0; auto row_node : mesh.getAllNodes())
    {
        const auto dof_cols = std::span{std::next(entries.get(), node_offsets[local_node_ind]),
                                        std::next(entries.get(), node_offsets[local_node_ind + 1])};
        for (auto dof_row : node_to_dof_map(row_node) | std::views::filter([](global_dof_t dof) {
                                return dof != NodeToGlobalDofMap< deduceNFields(problem_def) >::invalid_dof;
                            }))
            retval->insertGlobalIndices(dof_row, asTeuchosView(dof_cols));
        ++local_node_ind;
    }
    retval->endAssembly();
    return retval;
}
} // namespace lstr::detail
#endif // L3STER_ASSEMBLY_SPARSITYGRAPH_HPP
