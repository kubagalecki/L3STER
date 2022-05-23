#ifndef L3STER_UTILS_TRILINOSUTILS_HPP
#define L3STER_UTILS_TRILINOSUTILS_HPP

#include "l3ster/defs/Typedefs.h"

#include "Tpetra_CrsGraph.hpp"

#include <concepts>
#include <ranges>

namespace lstr
{
template < typename T, typename... Args >
Teuchos::RCP< T > makeTeuchosRCP(Args&&... args)
    requires std::constructible_from< T, Args... >
{
    return Teuchos::rcp(new T{std::forward< Args >(args)...});
}

template < std::predicate< global_dof_t > RowPred, std::predicate< global_dof_t > ColPred >
Teuchos::RCP< const Tpetra::CrsGraph<> >
getSubgraph(const Teuchos::RCP< const Tpetra::CrsGraph<> >& full_graph, RowPred&& is_row, ColPred&& is_col)
{
    auto get_row_cols = [&,
                         row_cols = std::vector< global_dof_t >(full_graph->getNodeMaxNumRowEntries()),
                         row_size = size_t{}](global_dof_t global_row) mutable {
        full_graph->getGlobalRowCopy(global_row, row_cols, row_size);
        return row_cols | std::views::take(row_size);
    };
    const auto foreach_row = [&](std::invocable< local_dof_t, global_dof_t > auto&& fun) {
        const auto signed_num_rows = static_cast< local_dof_t >(full_graph->getNodeNumRows());
        for (local_dof_t local_row = 0; local_row < signed_num_rows; ++local_row)
        {
            const auto global_row = full_graph->getRowMap()->getGlobalElement(local_row);
            fun(local_row, global_row);
        }
    };

    Kokkos::DualView< size_t* > subgraph_row_sizes_dv{"", full_graph->getNodeNumRows()};
    subgraph_row_sizes_dv.modify_host();
    auto subgraph_row_sizes = subgraph_row_sizes_dv.view_host();
    foreach_row([&](local_dof_t local_row, global_dof_t global_row) {
        if (is_row(global_row))
        {
            const auto row_cols           = get_row_cols(global_row);
            subgraph_row_sizes[local_row] = std::ranges::count_if(row_cols, is_col);
        }
        else
            subgraph_row_sizes[local_row] = 0;
    });
    subgraph_row_sizes_dv.sync_device();

    auto subgraph = makeTeuchosRCP< Tpetra::CrsGraph<> >(full_graph->getRowMap(), subgraph_row_sizes_dv);
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
