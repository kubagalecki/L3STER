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

using dim_t        = std::uint_fast8_t;                     // spatial dimension type
using n_id_t       = std::uint64_t;                         // node id type
using el_id_t      = std::uint64_t;                         // element id type
using el_o_t       = std::uint8_t;                          // element order type
using el_side_t    = std::uint8_t;                          // index of element side (points/edges/faces) type
using el_locind_t  = std::uint16_t;                         // local element index: nodes/element must not overflow this
using d_id_t       = std::uint16_t;                         // domain id type
using q_o_t        = std::uint8_t;                          // quadrature order type
using q_l_t        = std::uint64_t;                         // quadrature length type
using poly_o_t     = std::uint64_t;                         // polynomial order type
using val_t        = Tpetra::Vector<>::scalar_type;         // floating point value type
using global_dof_t = Tpetra::Vector<>::global_ordinal_type; // global DOF type
using local_dof_t  = Tpetra::Vector<>::local_ordinal_type;  // local DOF type
} // namespace lstr
#endif // L3STER_DEFS_TYPEDEFS_H
