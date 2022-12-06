#ifndef L3STER_BCS_DIRICHLETBC_HPP
#define L3STER_BCS_DIRICHLETBC_HPP

#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Vector.hpp"

#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

#include <ranges>
#include <span>

namespace lstr
{
namespace detail
{
template < typename T >
concept GlobalDofRange_c =
    std::ranges::sized_range< T > and std::same_as< std::ranges::range_value_t< T >, global_dof_t >;
} // namespace detail

class DirichletBCAlgebraic
{
public:
    using graph_t  = Tpetra::CrsGraph< local_dof_t, global_dof_t >;
    using matrix_t = Tpetra::CrsMatrix< val_t, local_dof_t, global_dof_t >;
    using vector_t = Tpetra::Vector< val_t, local_dof_t, global_dof_t >;

    template < detail::GlobalDofRange_c R1, detail::GlobalDofRange_c R2 >
    DirichletBCAlgebraic(const Teuchos::RCP< const graph_t >& sparsity_graph,
                         R1&&                                 owned_bc_dofs_sorted,
                         R2&&                                 shared_bc_dofs_sorted);

    inline void apply(const vector_t& bc_vals, matrix_t& matrix, vector_t& rhs) const;

private:
    [[nodiscard]] inline std::span< const local_dof_t > getLocalColInds(local_dof_t local_row) const;

    Teuchos::RCP< matrix_t >   m_dirichlet_col_mat;
    std::vector< local_dof_t > m_owned_bc_dofs, m_bc_local_col_inds, m_bc_local_col_inds_offsets;
};

template < detail::GlobalDofRange_c R1, detail::GlobalDofRange_c R2 >
DirichletBCAlgebraic::DirichletBCAlgebraic(const Teuchos::RCP< const graph_t >& sparsity_graph,
                                           R1&&                                 bc_rows,
                                           R2&&                                 bc_cols)
{
    const auto is_bc_row = [&](global_dof_t dof) {
        return std::ranges::binary_search(bc_rows, dof);
    };
    const auto is_bc_col = [&](global_dof_t dof) {
        return std::ranges::binary_search(bc_cols, dof);
    };
    m_dirichlet_col_mat = makeTeuchosRCP< matrix_t >(getSubgraph(
        sparsity_graph, [&](global_dof_t row) { return not is_bc_row(row); }, is_bc_col));

    m_owned_bc_dofs.reserve(std::ranges::size(bc_rows));
    m_bc_local_col_inds_offsets.reserve(sparsity_graph->getNodeNumRows() + 1);
    m_bc_local_col_inds_offsets.push_back(0);
    m_bc_local_col_inds.reserve(m_bc_local_col_inds_offsets.size()); // just a loose heuristic

    const auto process_dbc_row = [&](local_dof_t local_row) {
        m_owned_bc_dofs.push_back(local_row);
        m_bc_local_col_inds_offsets.push_back(m_bc_local_col_inds_offsets.back());
    };
    const auto process_non_dbc_row = [&](local_dof_t local_row) {
        const auto local_cols_full = getLocalRowView(*sparsity_graph, local_row);
        const auto local_cols_dbc  = getLocalRowView(*m_dirichlet_col_mat->getCrsGraph(), local_row);
        for (local_dof_t dbc_mat_col_local : local_cols_dbc)
        {
            const auto dbc_mat_col_global = m_dirichlet_col_mat->getColMap()->getGlobalElement(dbc_mat_col_local);
            const local_dof_t matched_local_row_ind =
                std::distance(local_cols_full.begin(), std::ranges::find_if(local_cols_full, [&](local_dof_t c) {
                                  return sparsity_graph->getColMap()->getGlobalElement(c) == dbc_mat_col_global;
                              }));
            m_bc_local_col_inds.push_back(matched_local_row_ind);
        }
        m_bc_local_col_inds_offsets.push_back(m_bc_local_col_inds_offsets.back() + local_cols_dbc.size());
    };
    for (local_dof_t local_row = 0; static_cast< size_t >(local_row) < sparsity_graph->getNodeNumRows(); ++local_row)
    {
        const auto global_row = sparsity_graph->getRowMap()->getGlobalElement(local_row);
        if (is_bc_row(global_row))
            process_dbc_row(local_row);
        else
            process_non_dbc_row(local_row);
    }
    m_bc_local_col_inds.shrink_to_fit();
}

void DirichletBCAlgebraic::apply(const vector_t& bc_vals, matrix_t& matrix, vector_t& rhs) const
{
    const auto matrix_row_map   = matrix.getRowMap();
    const auto matrix_col_map   = matrix.getColMap();
    const auto matrix_crs_graph = matrix.getCrsGraph();
    const auto bc_vals_data     = bc_vals.getData();

    rhs.sync_host();
    rhs.modify_host();
    auto rhs_data = rhs.getDataNonConst();

    std::vector< val_t >       local_vals(matrix.getLocalMaxNumRowEntries(), 0.);
    std::vector< val_t >       col_copy_vals(m_dirichlet_col_mat->getLocalMaxNumRowEntries());
    std::vector< local_dof_t > cols_to_zero_out(col_copy_vals.size());

    const auto process_dbc_row = [&](local_dof_t local_row) {
        const auto local_cols = getLocalRowView(*matrix_crs_graph, local_row);
        const auto global_row = matrix_row_map->getGlobalElement(local_row);
        const auto diag_ind   = std::distance(local_cols.begin(), std::ranges::find_if(local_cols, [&](local_dof_t c) {
                                                return matrix_col_map->getGlobalElement(c) == global_row;
                                            }));
        local_vals[diag_ind]  = 1.;
        replaceLocalValues(matrix, local_row, local_cols, local_vals | std::views::take(local_cols.size()));
        local_vals[diag_ind] = 0.;
        rhs_data[local_row]  = bc_vals_data[local_row];
    };
    const auto fill_subgraph_row = [&](local_dof_t                              local_row,
                                       const std::span< const local_dof_t >&    copy_inds,
                                       const Teuchos::ArrayView< const val_t >& local_vals_view) {
        std::ranges::transform(copy_inds, col_copy_vals.begin(), [&](local_dof_t i) { return local_vals_view[i]; });
        const auto copy_cols = getLocalRowView(*m_dirichlet_col_mat->getCrsGraph(), local_row);
        replaceLocalValues(
            *m_dirichlet_col_mat, local_row, copy_cols, col_copy_vals | std::views::take(copy_inds.size()));
    };
    const auto zero_dirichlet_cols = [&](local_dof_t                                    local_row,
                                         const std::span< const local_dof_t >&          copy_inds,
                                         const Teuchos::ArrayView< const local_dof_t >& local_cols) {
        std::ranges::transform(copy_inds, cols_to_zero_out.begin(), [&](local_dof_t i) { return local_cols[i]; });
        const auto replace_size = copy_inds.size();
        replaceLocalValues(matrix,
                           local_row,
                           cols_to_zero_out | std::views::take(replace_size),
                           local_vals | std::views::take(replace_size));
    };
    const auto process_non_dbc_row = [&](local_dof_t local_row) {
        const auto copycol_inds  = getLocalColInds(local_row);
        const auto local_entries = getLocalRowView(matrix, local_row);
        fill_subgraph_row(local_row, copycol_inds, local_entries.second);
        zero_dirichlet_cols(local_row, copycol_inds, local_entries.first);
    };

    m_dirichlet_col_mat->resumeFill();
    for (local_dof_t local_row = 0; static_cast< size_t >(local_row) < matrix.getLocalNumRows(); ++local_row)
    {
        if (std::ranges::binary_search(m_owned_bc_dofs, local_row))
            process_dbc_row(local_row);
        else
            process_non_dbc_row(local_row);
    }
    m_dirichlet_col_mat->fillComplete();
    rhs.sync_device();
    m_dirichlet_col_mat->apply(bc_vals, rhs, Teuchos::NO_TRANS, -1., 1.);
}

std::span< const local_dof_t > DirichletBCAlgebraic::getLocalColInds(local_dof_t local_row) const
{
    const auto col_inds_offs = m_bc_local_col_inds_offsets | std::views::drop(local_row) | std::views::take(2);
    return m_bc_local_col_inds | std::views::take(col_inds_offs[1]) | std::views::drop(col_inds_offs[0]);
}
} // namespace lstr
#endif // L3STER_BCS_DIRICHLETBC_HPP
