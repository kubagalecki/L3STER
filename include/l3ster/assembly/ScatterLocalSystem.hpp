#ifndef L3STER_ASSEMBLY_SCATTERLOCALSYSTEM_HPP
#define L3STER_ASSEMBLY_SCATTERLOCALSYSTEM_HPP

#include "l3ster/assembly/AssembleLocalSystem.hpp"
#include "l3ster/assembly/SparsityGraph.hpp"

#include <atomic>

namespace lstr::detail
{
// Note: Tpetra (Multi)Vector's sumIntoX member functions may throw in a multithreaded environment. This is due to the
// semantics of the dual views underpinning this class, not neccessarily to a bug. For the sake of simplicity, instead
// of figuring out the correct set of Kokkos views needed to make this work, here we pass in the underlying host
// allocation view (std::span) and perform the atomic update ourselves. Note that the sumIntoX member functions of
// Tpetra::CrsMatrix work fine, since Tpetra::CrsMatrix assumes modification in host space
template < int local_size >
void scatterLocalSystem(const eigen::RowMajorSquareMatrix< val_t, local_size >& local_matrix,
                        const Eigen::Vector< val_t, local_size >&               local_vector,
                        tpetra_crsmatrix_t&                                     global_matrix,
                        std::span< val_t >                                      global_vector,
                        std::span< const local_dof_t, size_t{local_size} >      row_dofs,
                        std::span< const local_dof_t, size_t{local_size} >      col_dofs,
                        std::span< const local_dof_t, size_t{local_size} >      rhs_dofs)
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
        const auto row_vals = std::span{std::next(local_matrix.data(), loc_row * size), static_cast< size_t >(size)};
        global_matrix.sumIntoLocalValues(row_dofs[loc_row], asTeuchosView(col_dofs), asTeuchosView(row_vals));
        std::atomic_ref{global_vector[rhs_dofs[loc_row]]}.fetch_add(local_vector[loc_row], std::memory_order_relaxed);
    }
}
} // namespace lstr::detail
#endif // L3STER_ASSEMBLY_SCATTERLOCALSYSTEM_HPP
