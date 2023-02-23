// Definition of data types used in L3STER

#ifndef L3STER_DEFS_TYPEDEFS_H
#define L3STER_DEFS_TYPEDEFS_H

#include "Tpetra_FECrsMatrix.hpp"
#include "Tpetra_FEMultiVector.hpp"

#include <cstdint>

namespace lstr
{
using std::ptrdiff_t;
using std::size_t;

using dim_t        = std::uint_fast8_t;                  // spatial dimension type
using n_id_t       = std::uint64_t;                      // node id type
using el_id_t      = std::uint64_t;                      // element id type
using el_o_t       = std::uint8_t;                       // element order type
using el_side_t    = std::uint8_t;                       // index of element side (points/edges/faces) type
using el_locind_t  = std::uint16_t;                      // local element index: nodes/element must not overflow this
using d_id_t       = std::uint16_t;                      // domain id type
using q_o_t        = std::uint8_t;                       // quadrature order type
using q_l_t        = std::uint64_t;                      // quadrature length type
using val_t        = Tpetra::MultiVector<>::scalar_type; // floating point value type
using global_dof_t = Tpetra::Map<>::global_ordinal_type; // global DOF type
using local_dof_t  = Tpetra::Map<>::local_ordinal_type;  // local DOF type

// For now use defaults, this may change in the future
using tpetra_crsgraph_t      = Tpetra::CrsGraph<>;
using tpetra_crsmatrix_t     = Tpetra::CrsMatrix<>;
using tpetra_fecrsgraph_t    = Tpetra::FECrsGraph<>;
using tpetra_fecrsmatrix_t   = Tpetra::FECrsMatrix<>;
using tpetra_femultivector_t = Tpetra::FEMultiVector<>;
using tpetra_import_t        = Tpetra::Import<>;
using tpetra_map_t           = Tpetra::Map<>;
using tpetra_multivector_t   = Tpetra::MultiVector<>;
using tpetra_operator_t      = Tpetra::Operator<>;
using tpetra_vector_t        = Tpetra::Vector<>;
} // namespace lstr
#endif // L3STER_DEFS_TYPEDEFS_H
