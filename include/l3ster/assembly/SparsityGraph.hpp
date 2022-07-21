#ifndef L3STER_ASSEMBLY_SPARSITYGRAPH_HPP
#define L3STER_ASSEMBLY_SPARSITYGRAPH_HPP

#include "l3ster/assembly/MakeTpetraMap.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/IndexMap.hpp"

#include "Tpetra_FECrsMatrix.hpp"
#include "Tpetra_FEMultiVector.hpp"

#include "tbb/tbb.h"

#include <mutex>

namespace lstr::detail
{
template < array_of< size_t > auto dof_inds, size_t n_nodes, size_t n_total_dofs >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >& nodes, const NodeToDofMap< n_total_dofs >& map)
{
    std::array< global_dof_t, dof_inds.size() * n_nodes > retval;
    for (auto insert_it = retval.begin(); auto node : nodes)
        insert_it = copyValuesAtInds< dof_inds >(map(node), insert_it);
    return retval;
}

template < array_of< size_t > auto dof_inds, ElementTypes T, el_o_t O, size_t n_total_dofs >
auto getSortedElementDofs(const Element< T, O >& element, const NodeToDofMap< n_total_dofs >& map)
{
    auto nodes_copy = element.getNodes();
    std::ranges::sort(nodes_copy);
    return getDofsFromNodes< dof_inds >(nodes_copy, map);
}

template < array_of< size_t > auto dof_inds, ElementTypes T, el_o_t O, size_t n_total_dofs >
auto getUnsortedElementDofs(const Element< T, O >& element, const NodeToDofMap< n_total_dofs >& map)
{
    return getDofsFromNodes< dof_inds >(element.getNodes(), map);
}

class CrsEntries
{
    static constexpr size_t buf_size        = 1ul << 29;
    static constexpr size_t overflowed_size = std::numeric_limits< size_t >::max();

    [[nodiscard]] global_dof_t*       getBuf(size_t row) noexcept { return std::next(buf.get(), row * row_buf_size); }
    [[nodiscard]] const global_dof_t* getBuf(size_t row) const noexcept
    {
        return std::next(buf.get(), row * row_buf_size);
    }

public:
    CrsEntries(size_t n_rows_)
        : buf{std::make_unique_for_overwrite< global_dof_t[] >(buf_size)},
          buf_sizes{std::make_unique< size_t[] >(n_rows_)},
          overflowed{std::make_unique< std::vector< global_dof_t >[] >(n_rows_)},
          row_buf_size{buf_size / (n_rows_ == 0 ? size_t{1} : n_rows_)},
          n_rows{n_rows_}
    {}

    [[nodiscard]] std::span< const global_dof_t > getRowEntries(size_t row) const noexcept
    {
        const auto row_size      = buf_sizes[row];
        const auto row_buf_start = getBuf(row);
        if (row_size != overflowed_size) [[likely]]
            return {row_buf_start, row_size};
        else [[unlikely]]
            return {overflowed[row].data(), overflowed[row].size()};
    }
    [[nodiscard]] std::span< global_dof_t > getRowEntriesForOverwrite(size_t row, size_t requested_size)
    {
        if (requested_size <= row_buf_size) [[likely]]
        {
            buf_sizes[row] = requested_size;
            return {getBuf(row), requested_size};
        }
        else [[unlikely]]
        {
            buf_sizes[row] = overflowed_size;
            overflowed[row].resize(requested_size);
            return {overflowed[row].data(), requested_size};
        }
    }
    [[nodiscard]] size_t size() const noexcept { return n_rows; }

private:
    std::unique_ptr< global_dof_t[] >                buf;
    std::unique_ptr< size_t[] >                      buf_sizes;
    std::unique_ptr< std::vector< global_dof_t >[] > overflowed;
    const size_t                                     row_buf_size;
    const size_t                                     n_rows;
};

template < auto problem_def >
auto calculateCrsData(const MeshPartition&                                        mesh,
                      ConstexprValue< problem_def >                               problem_def_ctwrapper,
                      const node_interval_vector_t< deduceNFields(problem_def) >& dof_intervals,
                      std::vector< global_dof_t >                                 owned_plus_shared_dofs)
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

    const auto                node_to_dof_map         = NodeToDofMap{mesh, dof_intervals};
    const auto                global_to_local_dof_map = IndexMap{owned_plus_shared_dofs};
    std::vector< std::mutex > row_mutexes(owned_plus_shared_dofs.size());

    const auto process_domain = [&]< auto dom_def >(ConstexprValue< dom_def >)
    {
        constexpr auto  domain_id        = dom_def.first;
        constexpr auto& coverage         = dom_def.second;
        constexpr auto  covered_dof_inds = getTrueInds< coverage >();

        const auto process_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
            const auto element_dofs = getSortedElementDofs< covered_dof_inds >(element, node_to_dof_map);
            std::bitset< std::tuple_size_v< decltype(element_dofs) > > processed_rows;
            do
                for (size_t row_ind = 0; auto row : element_dofs)
                {
                    const auto       local_row = global_to_local_dof_map(row);
                    std::unique_lock lock{row_mutexes[local_row], std::defer_lock};
                    if (lock.try_lock())
                    {
                        merge_new_dofs(local_row, element_dofs);
                        processed_rows.set(row_ind);
                    }
                    ++row_ind;
                }
            while (not processed_rows.all());
        };
        mesh.visit(process_element, domain_id, std::execution::par);
    };
    forConstexpr(process_domain, problem_def_ctwrapper);
    dealloc_scratchpads();
    return row_entries;
}

inline Kokkos::DualView< size_t* > getRowSizes(const CrsEntries& entries)
{
    Kokkos::DualView< size_t* > retval{"sparse graph row sizes", entries.size()};
    auto                        host_view = retval.view_host();
    retval.modify_host();
    tbb::parallel_for(tbb::blocked_range< size_t >{0, entries.size()}, [&](const tbb::blocked_range< size_t >& range) {
        for (size_t row = range.begin(); row != range.end(); ++row)
            host_view(row) = entries.getRowEntries(row).size();
    });
    retval.sync_device();
    return retval;
}

template < detail::ProblemDef_c auto problem_def >
Teuchos::RCP< const Tpetra::FECrsGraph< local_dof_t, global_dof_t > >
makeSparsityGraph(const MeshPartition&                                        mesh,
                  ConstexprValue< problem_def >                               problemdef_ctwrapper,
                  const node_interval_vector_t< deduceNFields(problem_def) >& dof_intervals,
                  const MpiComm&                                              comm)
{
    auto owned_dofs = detail::getNodeDofs(mesh.getNodes(), dof_intervals);
    auto owned_map  = makeTpetraMap(owned_dofs, comm);
    auto owned_plus_shared_dofs =
        concatVectors(std::move(owned_dofs), detail::getNodeDofs(mesh.getGhostNodes(), dof_intervals));
    auto owned_plus_shared_map = makeTpetraMap(owned_plus_shared_dofs, comm);

    const auto row_entries = calculateCrsData(mesh, problemdef_ctwrapper, dof_intervals, owned_plus_shared_dofs);
    const auto row_sizes   = getRowSizes(row_entries);

    using graph_t = Tpetra::FECrsGraph< local_dof_t, global_dof_t >;
    auto retval   = makeTeuchosRCP< graph_t >(owned_map, owned_plus_shared_map, row_sizes);
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
} // namespace lstr::detail
#endif // L3STER_ASSEMBLY_SPARSITYGRAPH_HPP
