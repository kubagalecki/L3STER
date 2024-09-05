#ifndef L3STER_UTILS_TRILINOSUTILS_HPP
#define L3STER_UTILS_TRILINOSUTILS_HPP

#include "l3ster/common/TrilinosTypedefs.h"
#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/Concepts.hpp"

#include <concepts>
#include <ranges>
#include <span>

namespace lstr::util
{
namespace detail
{
template < typename T >
inline constexpr bool is_kokkos_view = false;
template < typename... Params >
inline constexpr bool is_kokkos_view< Kokkos::View< Params... > > = true;
} // namespace detail

template < typename T >
concept KokkosView_c = detail::is_kokkos_view< std::decay_t< T > >;

template < typename R >
auto asTeuchosView(R&& range)
    requires std::ranges::contiguous_range< R > and std::ranges::sized_range< R > and
             std::ranges::borrowed_range< decltype(range) >
{
    return Teuchos::ArrayView{std::ranges::data(range), std::ranges::ssize(range)};
}

template < Arithmetic_c T, typename Layout, typename... Params >
auto asSpan(const Kokkos::View< T*, Layout, Params... >& view) -> std::span< T >
    requires std::same_as< Layout, Kokkos::LayoutLeft > or std::same_as< Layout, Kokkos::LayoutRight >
{
    return {view.data(), view.extent(0)};
}

template < typename... Params >
auto flatten(const Kokkos::View< Params... >& view) -> std::span< typename Kokkos::View< Params... >::value_type >
{
    return {view.data(), view.span()};
}

template < Arithmetic_c T, typename Layout, typename... Params >
auto asSpans(const Kokkos::View< T**, Layout, Params... >& view) -> std::vector< std::span< T > >
    requires std::same_as< Layout, Kokkos::LayoutLeft >
{
    const auto n_rows = view.extent(0), n_cols = view.extent(1);
    auto       retval = std::vector< std::span< T > >{};
    retval.reserve(n_cols);
    for (size_t i = 0; i != n_cols; ++i)
        retval.emplace_back(&view(0, i), n_rows);
    return retval;
}

template < size_t N, Arithmetic_c T, typename Layout, typename... Params >
auto asSpans(const Kokkos::View< T**, Layout, Params... >& view,
             std::integral_constant< size_t, N > = {}) -> std::array< std::span< T >, N >
    requires std::same_as< Layout, Kokkos::LayoutLeft >
{
    const auto n_rows = view.extent(0), n_cols = view.extent(1);
    throwingAssert(n_cols == N);
    auto retval = std::array< std::span< T >, N >{};
    for (size_t i = 0; i != n_cols; ++i)
        retval[i] = std::span{&view(0, i), n_rows};
    return retval;
}

// A std::span::subspan-like interface for a 1D Kokkos::View
template < typename T, typename... Args >
auto subview1D(const Kokkos::View< T*, Args... >& v, size_t offset, size_t count = std::dynamic_extent)
{
    return Kokkos::subview(v, std::pair{offset, count == std::dynamic_extent ? v.extent(0) : (offset + count)});
}

template < typename T, typename... Args >
auto makeTeuchosRCP(Args&&... args) -> Teuchos::RCP< T >
    requires std::constructible_from< T, decltype(std::forward< Args >(args))... >
{
    return Teuchos::rcp(new T(std::forward< Args >(args)...));
}

inline auto getLocalRowView(const tpetra_crsgraph_t& graph,
                            local_dof_t              row) -> tpetra_crsgraph_t::local_inds_host_view_type
{
    auto retval = tpetra_crsgraph_t::local_inds_host_view_type{};
    graph.getLocalRowView(row, retval);
    return retval;
}

inline auto getLocalRowView(const tpetra_crsmatrix_t& matrix, local_dof_t row)
{
    using local_inds_t = tpetra_crsmatrix_t::local_inds_host_view_type;
    using host_vals_t  = tpetra_crsmatrix_t::values_host_view_type;
    auto retval        = std::pair< local_inds_t, host_vals_t >{};
    matrix.getLocalRowView(row, retval.first, retval.second);
    return retval;
}

template < std::predicate< global_dof_t > RowPred, std::predicate< global_dof_t > ColPred >
auto getSubgraph(const Teuchos::RCP< const tpetra_crsgraph_t >& full_graph,
                 RowPred&&                                      is_row,
                 ColPred&&                                      is_col) -> Teuchos::RCP< tpetra_crsgraph_t >
{
    const auto full_num_rows   = full_graph->getLocalNumRows();
    const auto full_row_map    = full_graph->getRowMap();
    const auto full_max_row_sz = full_graph->getLocalMaxNumRowEntries();
    auto       get_row_cols    = [&,
                         row_cols = tpetra_crsgraph_t::nonconst_global_inds_host_view_type("row_cols", full_max_row_sz),
                         row_size = 0uz](global_dof_t global_row) mutable {
        full_graph->getGlobalRowCopy(global_row, row_cols, row_size);
        return std::span{row_cols.data(), row_size};
    };
    const auto foreach_row = [&]< std::invocable< local_dof_t, global_dof_t > F >(F&& fun) {
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

    auto retval = makeTeuchosRCP< tpetra_crsgraph_t >(full_row_map, subgraph_row_sizes_dv);
    foreach_row([&](local_dof_t local_row, global_dof_t global_row) {
        if (subgraph_row_sizes[local_row] == 0)
            return;
        auto       row_cols = get_row_cols(global_row);
        const auto subcols_end =
            std::ranges::remove_if(row_cols, [&](global_dof_t dof) { return not is_col(dof); }).begin();
        const Teuchos::ArrayView subcols{row_cols.data(), std::distance(row_cols.begin(), subcols_end)};
        retval->insertGlobalIndices(global_row, subcols);
    });
    retval->fillComplete();
    return retval;
}
} // namespace lstr::util
#endif // L3STER_UTILS_TRILINOSUTILS_HPP
