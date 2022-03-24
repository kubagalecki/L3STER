#ifndef L3STER_ASSEMBLY_ALGEBRAICSYSTEM_HPP
#define L3STER_ASSEMBLY_ALGEBRAICSYSTEM_HPP

#include "l3ster/global_assembly/MakeTpetraMap.hpp"

#include "Tpetra_FECrsMatrix.hpp"
#include "Tpetra_FEMultiVector.hpp"

namespace lstr
{
class AlgebraicSystem
{
public:
    using matrix_t = Tpetra::FECrsMatrix< val_t, local_dof_t, global_dof_t >;
    using vector_t = Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t >;

    template < size_t n_fields >
    AlgebraicSystem(const MeshPartition&                              partition,
                    const detail::node_interval_vector_t< n_fields >& dof_intervals,
                    const MpiComm&                                    comm);

private:
    Teuchos::RCP< matrix_t > matrix;
    Teuchos::RCP< vector_t > vector;
};

template < size_t n_fields >
AlgebraicSystem::AlgebraicSystem(const MeshPartition&                              partition,
                                 const detail::node_interval_vector_t< n_fields >& dof_intervals,
                                 const MpiComm&                                    comm)
{
    const auto owned_dofs             = detail::getNodeDofs(partition.getNodes(), dof_intervals);
    const auto shared_dofs            = detail::getNodeDofs(partition.getGhostNodes(), dof_intervals);
    const auto owned_plus_shared_dofs = concatVectors(owned_dofs, shared_dofs);

    auto owned_map             = makeTpetraMap(owned_dofs, comm);
    auto owned_plus_shared_map = makeTpetraMap(owned_plus_shared_dofs, comm);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ALGEBRAICSYSTEM_HPP
