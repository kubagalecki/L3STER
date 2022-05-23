#ifndef L3STER_CONTRIBUTELOCALSYSTEM_HPP
#define L3STER_CONTRIBUTELOCALSYSTEM_HPP

#include "l3ster/global_assembly/SparsityGraph.hpp"
#include "l3ster/local_assembly/AssembleLocalSystem.hpp"

namespace lstr
{
template < array_of< ptrdiff_t > auto dof_inds, int local_size, ElementTypes T, el_o_t O, size_t n_total_dofs >
void contributeLocalSystem(const std::pair< Eigen::Matrix< val_t, local_size, local_size, Eigen::RowMajor >,
                                            Eigen::Matrix< val_t, local_size, 1 > >&                    local_system,
                           const Element< T, O >&                                                       element,
                           const NodeToDofMap< n_total_dofs >&                                          map,
                           const Teuchos::RCP< Tpetra::CrsMatrix< val_t, local_dof_t, global_dof_t > >& global_matrix,
                           const Teuchos::RCP< Tpetra::Vector< val_t, local_dof_t, global_dof_t > >&    global_rhs)
    requires(dof_inds.size() * Element< T, O >::n_nodes == local_size)
{
    const auto& [local_matrix, local_rhs] = local_system;
    const auto el_dofs                    = detail::getUnsortedElementDofs< dof_inds >(element, map);
    const auto el_dofs_view               = Teuchos::ArrayView{el_dofs.data(), local_size};
    for (global_dof_t dof_ind = 0; auto dof : el_dofs)
    {
        const auto row_vals_view = Teuchos::ArrayView{std::next(local_matrix.data(), dof_ind * local_size), local_size};
        global_matrix->sumIntoGlobalValues(dof, el_dofs_view, row_vals_view);
        global_rhs->sumIntoGlobalValue(dof, local_rhs[dof_ind++]);
    }
}
} // namespace lstr
#endif // L3STER_CONTRIBUTELOCALSYSTEM_HPP
