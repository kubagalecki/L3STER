#ifndef L3STER_BCS_DIRICHLETBC_HPP
#define L3STER_BCS_DIRICHLETBC_HPP

#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

#include <ranges>
#include <span>

namespace lstr::bcs
{
class DirichletBCAlgebraic
{
public:
    template < SizedRangeOf_c< global_dof_t > Owned, SizedRangeOf_c< global_dof_t > Shared >
    DirichletBCAlgebraic(const Teuchos::RCP< const tpetra_crsgraph_t >& sparsity_graph,
                         Owned&&                                        owned_bc_dofs_sorted,
                         Shared&&                                       shared_bc_dofs_sorted);

    inline void apply(const tpetra_multivector_t& bc_vals, tpetra_crsmatrix_t& matrix, tpetra_multivector_t& rhs) const;

private:
    inline auto getLocalColInds(local_dof_t local_row) const -> std::span< const local_dof_t >;

    Teuchos::RCP< tpetra_crsmatrix_t > m_dirichlet_col_mat;
    std::vector< local_dof_t >         m_owned_bc_dofs, m_bc_local_col_inds, m_bc_local_col_inds_offsets;
};

template < SizedRangeOf_c< global_dof_t > Owned, SizedRangeOf_c< global_dof_t > Shared >
DirichletBCAlgebraic::DirichletBCAlgebraic(const Teuchos::RCP< const tpetra_crsgraph_t >& sparsity_graph,
                                           Owned&&                                        bc_rows,
                                           Shared&&                                       bc_cols)
{
    const auto& row_map   = *sparsity_graph->getRowMap();
    const auto& col_map   = *sparsity_graph->getColMap();
    const auto  is_bc_row = [&](global_dof_t dof) {
        return std::ranges::binary_search(bc_rows, dof);
    };
    const auto is_bc_col = [&](global_dof_t dof) {
        return std::ranges::binary_search(bc_cols, dof);
    };
    m_dirichlet_col_mat = util::makeTeuchosRCP< tpetra_crsmatrix_t >(util::getSubgraph(
        sparsity_graph, [&](global_dof_t row) { return not is_bc_row(row); }, is_bc_col));

    m_owned_bc_dofs.reserve(std::ranges::size(bc_rows));
    m_bc_local_col_inds_offsets.reserve(sparsity_graph->getLocalNumRows() + 1);
    m_bc_local_col_inds_offsets.push_back(0);
    m_bc_local_col_inds.reserve(m_bc_local_col_inds_offsets.size()); // just a loose heuristic

    const auto process_dbc_row = [&](local_dof_t local_row) {
        m_owned_bc_dofs.push_back(local_row);
        m_bc_local_col_inds_offsets.push_back(m_bc_local_col_inds_offsets.back());
    };
    const auto process_non_dbc_row = [&](local_dof_t local_row) {
        const auto local_cols_full = util::getLocalRowView(*sparsity_graph, local_row);
        const auto local_cols_dbc  = util::getLocalRowView(*m_dirichlet_col_mat->getCrsGraph(), local_row);
        for (size_t lc_ind = 0; lc_ind != local_cols_dbc.size(); ++lc_ind)
        {
            const auto dbc_mat_col_local  = local_cols_dbc[lc_ind];
            const auto dbc_mat_col_global = m_dirichlet_col_mat->getColMap()->getGlobalElement(dbc_mat_col_local);
            for (local_dof_t i = 0; i != std::ranges::ssize(local_cols_full); ++i)
                if (col_map.getGlobalElement(local_cols_full[i]) == dbc_mat_col_global)
                {
                    m_bc_local_col_inds.push_back(i);
                    break;
                }
        }
        m_bc_local_col_inds_offsets.push_back(m_bc_local_col_inds_offsets.back() +
                                              static_cast< local_dof_t >(local_cols_dbc.size()));
    };
    for (local_dof_t local_row = 0; static_cast< size_t >(local_row) < sparsity_graph->getLocalNumRows(); ++local_row)
    {
        const auto global_row = row_map.getGlobalElement(local_row);
        if (is_bc_row(global_row))
            process_dbc_row(local_row);
        else
            process_non_dbc_row(local_row);
    }
    m_bc_local_col_inds.shrink_to_fit();
}

void DirichletBCAlgebraic::apply(const tpetra_multivector_t& bc_vals,
                                 tpetra_crsmatrix_t&         matrix,
                                 tpetra_multivector_t&       rhs) const
{
    util::throwingAssert(bc_vals.getNumVectors() == rhs.getNumVectors(),
                         "Number of RHSs must be equal to the number of prescribed values");
    {
        const auto& matrix_row_map   = *matrix.getRowMap();
        const auto& matrix_col_map   = *matrix.getColMap();
        const auto& matrix_crs_graph = *matrix.getCrsGraph();

        const auto bc_vals_view  = bc_vals.getLocalViewHost(Tpetra::Access::ReadOnly);
        const auto rhs_view      = rhs.getLocalViewHost(Tpetra::Access::ReadWrite);
        const auto n_rhs         = rhs_view.extent(1);
        auto       col_copy_vals = Kokkos::View< val_t* >{"Columns", m_dirichlet_col_mat->getLocalMaxNumRowEntries()};
        auto       cols_to_zero_out = Kokkos::View< local_dof_t* >{"Column inds to zero out", col_copy_vals.size()};
        auto       local_vals       = Kokkos::View< val_t* >{"Local values", matrix.getLocalMaxNumRowEntries()};
        for (size_t i = 0; i < local_vals.extent(0); ++i)
            local_vals[i] = 0.;

        const auto process_dbc_row = [&](local_dof_t local_row) {
            const auto local_cols = util::getLocalRowView(matrix_crs_graph, local_row);
            const auto global_row = matrix_row_map.getGlobalElement(local_row);
            const auto diag_ind =
                *std::ranges::find(std::views::iota(size_t{}, local_cols.extent(0)), global_row, [&](auto index) {
                    return matrix_col_map.getGlobalElement(local_cols[index]);
                });
            local_vals[diag_ind] = 1.;
            matrix.replaceLocalValues(local_row, local_cols, util::subview1D(local_vals, 0, local_cols.extent_int(0)));
            local_vals[diag_ind] = 0.;
            for (size_t i = 0; i != n_rhs; ++i)
                rhs_view(local_row, i) = bc_vals_view(local_row, i);
        };
        const auto fill_subgraph_row = [&](local_dof_t                           local_row,
                                           const std::span< const local_dof_t >& copy_inds,
                                           const Kokkos::View< const val_t* >&   local_vals_view) {
            std::ranges::transform(copy_inds, col_copy_vals.data(), [&](local_dof_t i) { return local_vals_view[i]; });
            const auto copy_cols = util::getLocalRowView(*m_dirichlet_col_mat->getCrsGraph(), local_row);
            m_dirichlet_col_mat->replaceLocalValues(
                local_row, copy_cols, util::subview1D(col_copy_vals, 0, copy_inds.size()));
        };
        const auto zero_dirichlet_cols = [&](local_dof_t                               local_row,
                                             const std::span< const local_dof_t >&     copy_inds,
                                             const Kokkos::View< const local_dof_t* >& local_cols) {
            std::ranges::transform(copy_inds, cols_to_zero_out.data(), [&](local_dof_t i) { return local_cols[i]; });
            const auto replace_size = copy_inds.size();
            matrix.replaceLocalValues(local_row,
                                      util::subview1D(cols_to_zero_out, 0, replace_size),
                                      util::subview1D(local_vals, 0, replace_size));
        };
        const auto process_non_dbc_row = [&](local_dof_t local_row) {
            const auto copycol_inds = getLocalColInds(local_row);
            const auto [inds, vals] = util::getLocalRowView(matrix, local_row);
            fill_subgraph_row(local_row, copycol_inds, vals);
            zero_dirichlet_cols(local_row, copycol_inds, inds);
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
    }
    m_dirichlet_col_mat->apply(bc_vals, rhs, Teuchos::NO_TRANS, -1., 1.);
}

auto DirichletBCAlgebraic::getLocalColInds(local_dof_t local_row) const -> std::span< const local_dof_t >
{
    const auto col_inds_offs = m_bc_local_col_inds_offsets | std::views::drop(local_row) | std::views::take(2);
    return m_bc_local_col_inds | std::views::take(col_inds_offs[1]) | std::views::drop(col_inds_offs[0]);
}
} // namespace lstr::bcs
#endif // L3STER_BCS_DIRICHLETBC_HPP
