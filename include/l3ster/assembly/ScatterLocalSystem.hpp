#ifndef L3STER_ASSEMBLY_SCATTERLOCALSYSTEM_HPP
#define L3STER_ASSEMBLY_SCATTERLOCALSYSTEM_HPP

#include "l3ster/assembly/AssembleLocalSystem.hpp"
#include "l3ster/assembly/SparsityGraph.hpp"

#include <atomic>

namespace lstr::detail
{
// Note: Tpetra (Multi)Vector's sumIntoX member functions often throw in a multithreaded environment. This is due to the
// semantics of the dual views underpinning this class, not neccessarily to a bug. For the sake of simplicity, instead
// of figuring out the correct set of Kokkos views needed to make this work, here we pass in the underlying host
// allocation and perform the atomic update ourselves. Note that the sumIntoX member functions of Tpetra::CrsMatrix work
// fine, since Tpetra::CrsMatrix assumes modification in host space
template < int local_size >
void scatterLocalVector(const Eigen::Matrix< val_t, local_size, 1 >&          local_vector,
                        const std::array< global_dof_t, size_t{local_size} >& dofs,
                        const Teuchos::ArrayRCP< val_t >&                     global_vector,
                        const Tpetra::Map< local_dof_t, global_dof_t >&       dof_map)
{
    for (ptrdiff_t local_row = 0; auto dof : dofs)
    {
        const auto local_dof = dof_map.getLocalElement(dof);
        std::atomic_ref{global_vector[local_dof]}.fetch_add(local_vector[local_row++], std::memory_order_relaxed);
    }
}

template < int local_size >
void scatterLocalMatrix(const Eigen::Matrix< val_t, local_size, local_size, Eigen::RowMajor >& local_matrix,
                        const std::array< global_dof_t, size_t{local_size} >&                  dofs,
                        Tpetra::CrsMatrix< val_t, local_dof_t, global_dof_t >&                 global_matrix)
{
    for (ptrdiff_t local_row = 0; auto dof : dofs)
    {
        const auto row_vals = std::views::counted(std::next(local_matrix.data(), local_row++ * local_size), local_size);
        global_matrix.sumIntoGlobalValues(dof, asTeuchosView(dofs), asTeuchosView(row_vals));
    }
}
} // namespace lstr::detail
#endif // L3STER_ASSEMBLY_SCATTERLOCALSYSTEM_HPP
