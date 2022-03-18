// Definition of data types used in L3STER

#ifndef L3STER_DEFS_TYPEDEFS_H
#define L3STER_DEFS_TYPEDEFS_H

#include <cstdint>

namespace lstr
{
using std::ptrdiff_t;
using std::size_t;

using val_t        = double;             // floating point value type
using coord_t      = double;             // floating point coordinate type
using dim_t        = std::uint_fast8_t;  // spatial dimension type
using n_id_t       = std::uint_fast64_t; // node id type
using el_id_t      = std::uint_fast64_t; // element id type
using el_o_t       = std::uint_fast8_t;  // element order type
using el_ns_t      = std::uint_fast8_t;  // number of element sides (points/edges/faces) type
using el_locind_t  = std::uint_fast16_t; // local element index t., #nodes/element must not overflow this
using d_id_t       = std::uint_fast8_t;  // domain id type
using q_o_t        = std::uint_fast8_t;  // quadrature order type
using q_l_t        = std::uint_fast64_t; // quadrature length type
using poly_o_t     = std::uint_fast64_t; // polynomial order type
using global_dof_t = std::int_fast64_t;  // global DOF type
} // namespace lstr
#endif // L3STER_DEFS_TYPEDEFS_H
