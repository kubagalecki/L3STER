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

private:
    Teuchos::RCP< matrix_t > matrix;
    Teuchos::RCP< vector_t > vector;
};
} // namespace lstr
#endif // L3STER_ASSEMBLY_ALGEBRAICSYSTEM_HPP
