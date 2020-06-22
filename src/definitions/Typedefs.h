// Definition of data types used in L3STER

#ifndef L3STER_INCGUARD_DEFINITIONS_H
#define L3STER_INCGUARD_DEFINITIONS_H

#include <cstddef>
#include <cstdint>

//  n    - node
//  el   - element
//  q    - quadrature
//  val  - value
//  d    - domain
//  poly - polynomial
//
//  id   - ID
//  o    - order
//  dim  - dimension
//  l    - length

namespace lstr::types
{
using val_t    = double;
using dim_t    = uint_fast8_t;
using n_id_t   = uint_fast64_t;
using el_id_t  = uint_fast8_t;
using el_o_t   = uint_fast8_t;
using d_id_t   = uint_fast8_t;
using q_o_t    = uint_fast8_t;
using q_l_t    = size_t;
using poly_o_t = size_t;
} // namespace lstr::types

#endif // L3STER_INCGUARD_DEFINITIONS_H
