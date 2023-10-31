#ifndef L3STER_GLOB_ASM_SCATTERLOCALSYSTEM_HPP
#define L3STER_GLOB_ASM_SCATTERLOCALSYSTEM_HPP

#include "l3ster/glob_asm/AssembleLocalSystem.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

#include <atomic>

namespace lstr::glob_asm
{
namespace detail
{
consteval size_t eigenToSpanSize(int eigen_size)
{
    return eigen_size == Eigen::Dynamic ? std::dynamic_extent : static_cast< size_t >(eigen_size);
}
} // namespace detail

// Note: Tpetra (Multi)Vector's sumIntoX member functions may throw in a multithreaded environment. This is due to the
// semantics of the dual views underpinning this class, not neccessarily to a bug. For the sake of simplicity, instead
// of figuring out the correct set of Kokkos views needed to make this work, here we pass in the underlying host
// allocation view (std::span) and perform the atomic update ourselves. Note that the sumIntoX member functions of
// Tpetra::CrsMatrix work fine, since Tpetra::CrsMatrix assumes modification in host space
template < int local_size, size_t n_rhs >
void scatterLocalSystem(const util::eigen::RowMajorSquareMatrix< val_t, local_size >&       local_matrix,
                        const Eigen::Matrix< val_t, local_size, int{n_rhs} >&               local_rhs,
                        tpetra_crsmatrix_t&                                                 global_matrix,
                        const std::array< std::span< val_t >, n_rhs >&                      global_rhs,
                        std::span< const local_dof_t, detail::eigenToSpanSize(local_size) > row_dofs,
                        std::span< const local_dof_t, detail::eigenToSpanSize(local_size) > col_dofs,
                        std::span< const local_dof_t, detail::eigenToSpanSize(local_size) > rhs_dofs)
{
    L3STER_PROFILE_FUNCTION;
    const auto size = std::invoke([&] {
        if constexpr (local_size == Eigen::Dynamic)
            return local_matrix.cols();
        else
            return local_size;
    });
    for (ptrdiff_t loc_row = 0; loc_row != size; ++loc_row)
    {
        const auto mat_row  = row_dofs[loc_row];
        const auto mat_cols = util::asTeuchosView(col_dofs);
        const auto mat_vals = Teuchos::ArrayView{std::next(local_matrix.data(), loc_row * size), size};
        global_matrix.sumIntoLocalValues(mat_row, mat_cols, mat_vals);

        const auto rhs_ind = rhs_dofs[loc_row];
        for (size_t i = 0; i != n_rhs; ++i)
        {
            const auto rhs_val = local_rhs(loc_row, i);
            std::atomic_ref{global_rhs[i][rhs_ind]}.fetch_add(rhs_val, std::memory_order_relaxed);
        }
    }
}
} // namespace lstr::glob_asm
#endif // L3STER_GLOB_ASM_SCATTERLOCALSYSTEM_HPP
