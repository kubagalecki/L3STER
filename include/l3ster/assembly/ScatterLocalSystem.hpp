#ifndef L3STER_ASSEMBLY_SCATTERLOCALSYSTEM_HPP
#define L3STER_ASSEMBLY_SCATTERLOCALSYSTEM_HPP

#include "l3ster/assembly/AssembleLocalSystem.hpp"
#include "l3ster/assembly/SparsityGraph.hpp"

namespace lstr
{
namespace detail
{
template < int local_size >
void scatterLocalVector(const Eigen::Matrix< val_t, local_size, 1 >&          local_vector,
                        const std::array< global_dof_t, size_t{local_size} >& dofs,
                        Tpetra::Vector< val_t, local_dof_t, global_dof_t >&   global_vector)
{
    for (ptrdiff_t local_row = 0; auto dof : dofs)
        global_vector.sumIntoGlobalValue(dof, local_vector[local_row++]);
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
} // namespace detail

template < array_of< size_t > auto dof_inds, int local_size, ElementTypes T, el_o_t O, size_t n_fields >
void scatterLocalSystem(const std::pair< Eigen::Matrix< val_t, local_size, local_size, Eigen::RowMajor >,
                                         Eigen::Matrix< val_t, local_size, 1 > >& local_system,
                        const Element< T, O >&                                    element,
                        const NodeToDofMap< n_fields >&                           map,
                        Tpetra::CrsMatrix< val_t, local_dof_t, global_dof_t >&    global_matrix,
                        Tpetra::Vector< val_t, local_dof_t, global_dof_t >&       global_rhs)
    requires(dof_inds.size() * Element< T, O >::n_nodes == local_size)
{
    static_assert(std::ranges::all_of(dof_inds, [](auto dof) { return dof < n_fields; }));

    const auto& [local_matrix, local_rhs] = local_system;
    const auto el_dofs                    = detail::getUnsortedElementDofs< dof_inds >(element, map);
    detail::scatterLocalVector(local_rhs, el_dofs, global_rhs);
    detail::scatterLocalMatrix(local_matrix, el_dofs, global_matrix);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_SCATTERLOCALSYSTEM_HPP
