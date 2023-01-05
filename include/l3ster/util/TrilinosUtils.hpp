#ifndef L3STER_UTILS_TRILINOSUTILS_HPP
#define L3STER_UTILS_TRILINOSUTILS_HPP

#include "l3ster/defs/Typedefs.h"

#include "Tpetra_CrsMatrix.hpp"

#include <concepts>
#include <ranges>
#include <span>

namespace lstr
{
auto asTeuchosView(std::ranges::contiguous_range auto&& range)
    requires std::ranges::sized_range< decltype(range) > and std::ranges::borrowed_range< decltype(range) >
{
    return Teuchos::ArrayView{std::ranges::data(range), std::ranges::ssize(range)};
}

template < Arithmetic_c T, typename Layout, typename... Params >
auto asSpan(const Kokkos::View< T*, Layout, Params... >& view) -> std::span< T >
    requires std::same_as< Layout, Kokkos::LayoutLeft > or std::same_as< Layout, Kokkos::LayoutRight >
{
    return {view.data(), view.extent(0)};
}

// A std::span::subspan-like interface for a 1D Kokkos::View
template < typename T, typename... Args >
auto subview1D(const Kokkos::View< T*, Args... >& v, size_t offset, size_t count = std::dynamic_extent)
{
    return Kokkos::subview(v, std::pair{offset, count != std::dynamic_extent ? (offset + count) : v.extent(0)});
}

template < typename T, typename... Args >
Teuchos::RCP< T > makeTeuchosRCP(Args&&... args)
    requires std::constructible_from< T, Args... >
{
    return Teuchos::rcp(new T{std::forward< Args >(args)...});
}

inline auto getLocalRowView(const tpetra_crsgraph_t& graph, local_dof_t row)
{
    tpetra_crsgraph_t::local_inds_host_view_type retval;
    graph.getLocalRowView(row, retval);
    return retval;
}

inline auto getLocalRowView(const tpetra_crsmatrix_t& matrix, local_dof_t row)
{
    std::pair< tpetra_crsmatrix_t::local_inds_host_view_type, tpetra_crsmatrix_t::values_host_view_type > retval;
    matrix.getLocalRowView(row, retval.first, retval.second);
    return retval;
}

auto getSubgraph(const Teuchos::RCP< const tpetra_crsgraph_t >& full_graph,
                 std::predicate< global_dof_t > auto&&          is_row,
                 std::predicate< global_dof_t > auto&&          is_col)
{
    const auto full_num_rows   = full_graph->getLocalNumRows();
    const auto full_row_map    = full_graph->getRowMap();
    const auto full_max_row_sz = full_graph->getLocalMaxNumRowEntries();
    auto       get_row_cols    = [&,
                         row_cols = tpetra_crsgraph_t::nonconst_global_inds_host_view_type("row_cols", full_max_row_sz),
                         row_size = size_t{}](global_dof_t global_row) mutable {
        full_graph->getGlobalRowCopy(global_row, row_cols, row_size);
        return std::span{row_cols.data(), row_size};
    };
    const auto foreach_row = [&](std::invocable< local_dof_t, global_dof_t > auto&& fun) {
        const auto signed_num_rows = static_cast< local_dof_t >(full_num_rows);
        for (local_dof_t local_row = 0; local_row < signed_num_rows; ++local_row)
        {
            const auto global_row = full_row_map->getGlobalElement(local_row);
            fun(local_row, global_row);
        }
    };

    Kokkos::DualView< size_t* > subgraph_row_sizes_dv{"", full_num_rows};
    subgraph_row_sizes_dv.modify_host();
    auto subgraph_row_sizes = subgraph_row_sizes_dv.view_host();
    foreach_row([&](local_dof_t local_row, global_dof_t global_row) {
        if (is_row(global_row))
            subgraph_row_sizes[local_row] = std::ranges::count_if(get_row_cols(global_row), is_col);
        else
            subgraph_row_sizes[local_row] = 0;
    });
    subgraph_row_sizes_dv.sync_device();

    auto subgraph = makeTeuchosRCP< tpetra_crsgraph_t >(full_row_map, subgraph_row_sizes_dv);
    foreach_row([&](local_dof_t local_row, global_dof_t global_row) {
        if (subgraph_row_sizes[local_row] == 0)
            return;
        auto       row_cols = get_row_cols(global_row);
        const auto subcols_end =
            std::ranges::remove_if(row_cols, [&](global_dof_t dof) { return not is_col(dof); }).begin();
        const Teuchos::ArrayView subcols{row_cols.data(), std::distance(row_cols.begin(), subcols_end)};
        subgraph->insertGlobalIndices(global_row, subcols);
    });
    subgraph->fillComplete();
    return subgraph;
}
} // namespace lstr
#endif // L3STER_UTILS_TRILINOSUTILS_HPP
