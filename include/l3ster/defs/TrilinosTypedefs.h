#ifndef L3STER_DEFS_TRILINOSTYPEDEFS_H
#define L3STER_DEFS_TRILINOSTYPEDEFS_H

#include "l3ster/defs/Typedefs.h"

#include <concepts>

// Disable diagnostics triggered by Trilinos
#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wvolatile"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#endif

#include "Kokkos_Core.hpp"
#include "Tpetra_FECrsMatrix.hpp"
#include "Tpetra_FEMultiVector.hpp"
#include "Tpetra_Vector.hpp"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

namespace lstr
{
static_assert(std::same_as< val_t, Tpetra::MultiVector<>::scalar_type >);
static_assert(std::same_as< local_dof_t, Tpetra::MultiVector<>::local_ordinal_type >);
static_assert(std::same_as< global_dof_t, Tpetra::MultiVector<>::global_ordinal_type >);

// For now use default node argument, this may change in the future (to support easy config for CPU/GPU)
using tpetra_crsgraph_t      = Tpetra::CrsGraph< local_dof_t, global_dof_t >;
using tpetra_crsmatrix_t     = Tpetra::CrsMatrix< val_t, local_dof_t, global_dof_t >;
using tpetra_fecrsgraph_t    = Tpetra::FECrsGraph< local_dof_t, global_dof_t >;
using tpetra_fecrsmatrix_t   = Tpetra::FECrsMatrix< val_t, local_dof_t, global_dof_t >;
using tpetra_femultivector_t = Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t >;
using tpetra_import_t        = Tpetra::Import< local_dof_t, global_dof_t >;
using tpetra_map_t           = Tpetra::Map< local_dof_t, global_dof_t >;
using tpetra_multivector_t   = Tpetra::MultiVector< val_t, local_dof_t, global_dof_t >;
using tpetra_operator_t      = Tpetra::Operator< val_t, local_dof_t, global_dof_t >;
using tpetra_vector_t        = Tpetra::Vector< val_t, local_dof_t, global_dof_t >;
} // namespace lstr
#endif // L3STER_DEFS_TRILINOSTYPEDEFS_H
