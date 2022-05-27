#ifndef L3STER_DIRICHLETBC_HPP
#define L3STER_DIRICHLETBC_HPP

#include "l3ster/util/TrilinosUtils.hpp"

#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Vector.hpp"

#include <ranges>
#include <span>

#include "l3ster/util/DynamicBitset.hpp"

namespace lstr
{
namespace detail
{
template < typename T >
concept GlobalDofRange_c = std::ranges::sized_range< T > and std::same_as < std::ranges::range_value_t< T >,
global_dof_t > ;
} // namespace detail

class DirichletBCAlgebraic
{
    [[nodiscard]] std::span< const local_dof_t > getLocalColInds(local_dof_t local_row) const
    {
        return std::span{std::next(bc_local_col_inds.cbegin(), bc_local_col_inds_offsets[local_row]),
                         std::next(bc_local_col_inds.cbegin(), bc_local_col_inds_offsets[local_row + 1])};
    }

public:
    template < detail::GlobalDofRange_c R1, detail::GlobalDofRange_c R2 >
    DirichletBCAlgebraic(const Teuchos::RCP< const Tpetra::CrsGraph<> >& dof,
                         R1&&                                            owned_bc_dofs_sorted,
                         R2&&                                            shared_bc_dofs_sorted);

    inline void apply(const Tpetra::Vector<>& bc_vals, Tpetra::CrsMatrix<>& matrix, Tpetra::Vector<>& rhs) const;

private:
    Teuchos::RCP< Tpetra::CrsMatrix<> > dirichlet_col_matrix;
    std::vector< local_dof_t >          owned_bc_dofs;
    std::vector< local_dof_t >          bc_local_col_inds;
    std::vector< local_dof_t >          bc_local_col_inds_offsets;
};

template < detail::GlobalDofRange_c R1, detail::GlobalDofRange_c R2 >
DirichletBCAlgebraic::DirichletBCAlgebraic(const Teuchos::RCP< const Tpetra::CrsGraph<> >& sparsity_graph,
                                           R1&&                                            bc_rows,
                                           R2&&                                            bc_cols)
{
    const auto is_bc_row = [&](global_dof_t dof) {
        return std::ranges::binary_search(bc_rows, dof);
    };
    const auto is_bc_col = [&](global_dof_t dof) {
        return std::ranges::binary_search(bc_cols, dof);
    };
    dirichlet_col_matrix = makeTeuchosRCP< Tpetra::CrsMatrix<> >(getSubgraph(
        sparsity_graph, [&](global_dof_t row) { return not is_bc_row(row); }, is_bc_col));

    owned_bc_dofs.reserve(std::ranges::size(bc_rows));
    bc_local_col_inds_offsets.reserve(sparsity_graph->getNodeNumRows() + 1);
    bc_local_col_inds_offsets.push_back(0);
    bc_local_col_inds.reserve(bc_local_col_inds_offsets.size()); // just a loose heuristic

    const auto process_dbc_row = [&](local_dof_t local_row) {
        owned_bc_dofs.push_back(local_row);
        bc_local_col_inds_offsets.push_back(bc_local_col_inds_offsets.back());
    };
    const auto process_non_dbc_row = [&](local_dof_t local_row) {
        Teuchos::ArrayView< const local_dof_t > local_cols_full, local_cols_dbc;
        sparsity_graph->getLocalRowView(local_row, local_cols_full);
        dirichlet_col_matrix->getCrsGraph()->getLocalRowView(local_row, local_cols_dbc);
        for (local_dof_t dbc_mat_col_local : local_cols_dbc)
        {
            const auto dbc_mat_col_global = dirichlet_col_matrix->getColMap()->getGlobalElement(dbc_mat_col_local);
            const local_dof_t matched_local_row_ind =
                std::distance(local_cols_full.begin(), std::ranges::find_if(local_cols_full, [&](local_dof_t c) {
                                  return sparsity_graph->getColMap()->getGlobalElement(c) == dbc_mat_col_global;
                              }));
            bc_local_col_inds.push_back(matched_local_row_ind);
        }
        bc_local_col_inds_offsets.push_back(bc_local_col_inds_offsets.back() + local_cols_dbc.size());
    };
    for (local_dof_t local_row = 0; static_cast< size_t >(local_row) < sparsity_graph->getNodeNumRows(); ++local_row)
    {
        const auto global_row = sparsity_graph->getRowMap()->getGlobalElement(local_row);
        if (is_bc_row(global_row))
            process_dbc_row(local_row);
        else
            process_non_dbc_row(local_row);
    }
    bc_local_col_inds.shrink_to_fit();
}

void DirichletBCAlgebraic::apply(const Tpetra::Vector<>& bc_vals,
                                 Tpetra::CrsMatrix<>&    matrix,
                                 Tpetra::Vector<>&       rhs) const
{
    std::vector< val_t >       local_vals(matrix.getNodeMaxNumRowEntries(), 0.);
    std::vector< val_t >       col_copy_vals(dirichlet_col_matrix->getNodeMaxNumRowEntries());
    std::vector< local_dof_t > cols_to_zero_out(col_copy_vals.size());

    const auto process_dbc_row = [&](local_dof_t local_row) {
        Teuchos::ArrayView< const local_dof_t > local_cols;
        matrix.getCrsGraph()->getLocalRowView(local_row, local_cols);
        const auto global_row = matrix.getRowMap()->getGlobalElement(local_row);
        const auto diag_ind   = std::distance(local_cols.begin(), std::ranges::find_if(local_cols, [&](local_dof_t c) {
                                                return matrix.getColMap()->getGlobalElement(c) == global_row;
                                            }));
        local_vals[diag_ind]  = 1.;
        const auto local_vals_view = Teuchos::ArrayView< val_t >{local_vals.data(), local_cols.size()};
        matrix.replaceLocalValues(local_row, local_cols, local_vals_view);
        local_vals[diag_ind] = 0.;
        rhs.replaceLocalValue(local_row, bc_vals.getData()[local_row]);
    };
    const auto get_local_row_view = [&](local_dof_t local_row) {
        std::pair< Teuchos::ArrayView< const local_dof_t >, Teuchos::ArrayView< const val_t > > retval;
        matrix.getLocalRowView(local_row, retval.first, retval.second);
        return retval;
    };
    const auto fill_subgraph_row = [&](local_dof_t                              local_row,
                                       const std::span< const local_dof_t >&    copy_inds,
                                       const Teuchos::ArrayView< const val_t >& local_vals_view) {
        std::ranges::transform(copy_inds, col_copy_vals.begin(), [&](local_dof_t i) { return local_vals_view[i]; });
        const auto copy_ssize = static_cast< Teuchos::ArrayView< val_t >::size_type >(copy_inds.size());
        const auto copy_view  = Teuchos::ArrayView{col_copy_vals.data(), copy_ssize};
        Teuchos::ArrayView< const local_dof_t > copy_cols;
        dirichlet_col_matrix->getCrsGraph()->getLocalRowView(local_row, copy_cols);
        dirichlet_col_matrix->replaceLocalValues(local_row, copy_cols, copy_view);
    };
    const auto zero_dirichlet_cols = [&](local_dof_t                                    local_row,
                                         const std::span< const local_dof_t >&          copy_inds,
                                         const Teuchos::ArrayView< const local_dof_t >& local_cols) {
        std::ranges::transform(copy_inds, cols_to_zero_out.begin(), [&](local_dof_t i) { return local_cols[i]; });
        const auto zero_ssize = static_cast< Teuchos::ArrayView< val_t >::size_type >(copy_inds.size());
        const Teuchos::ArrayView< const local_dof_t > cols_view{cols_to_zero_out.data(), zero_ssize};
        const Teuchos::ArrayView< const val_t >       vals_view{local_vals.data(), zero_ssize};
        matrix.replaceLocalValues(local_row, cols_view, vals_view);
    };
    const auto process_non_dbc_row = [&](local_dof_t local_row) {
        const auto copycol_inds  = getLocalColInds(local_row);
        const auto local_entries = get_local_row_view(local_row);
        fill_subgraph_row(local_row, copycol_inds, local_entries.second);
        zero_dirichlet_cols(local_row, copycol_inds, local_entries.first);
    };

    dirichlet_col_matrix->resumeFill();
    for (local_dof_t local_row = 0; static_cast< size_t >(local_row) < matrix.getNodeNumRows(); ++local_row)
    {
        if (std::ranges::binary_search(owned_bc_dofs, local_row))
            process_dbc_row(local_row);
        else
            process_non_dbc_row(local_row);
    }
    dirichlet_col_matrix->fillComplete();
    dirichlet_col_matrix->apply(bc_vals, rhs, Teuchos::NO_TRANS, -1., 1.);
}
} // namespace lstr
#endif // L3STER_DIRICHLETBC_HPP
