// Definition of data types used in L3STER

#ifndef L3STER_INCGUARD_DEFINITIONS_TYPEDEFS_H
#define L3STER_INCGUARD_DEFINITIONS_TYPEDEFS_H

#include <cstddef>
#include <cstdint>

namespace lstr::types
{
using val_t    = double;        // floating point value type
using coord_t  = double;        // floating point coordinate type
using dim_t    = uint_fast8_t;  // spatial dimension type
using n_id_t   = uint_fast64_t; // node id type
using el_id_t  = uint_fast64_t; // element id type
using el_o_t   = uint_fast8_t;  // element order type
using el_f_t   = uint_fast8_t;  // element face type (determines edge/face of boundary elements)
using d_id_t   = uint_fast8_t;  // domain id type
using q_o_t    = uint_fast8_t;  // quadrature order type
using q_l_t    = uint_fast64_t; // quadrature length type
using poly_o_t = uint_fast64_t; // polynomial order type
} // namespace lstr::types

#endif // L3STER_INCGUARD_DEFINITIONS_TYPEDEFS_H