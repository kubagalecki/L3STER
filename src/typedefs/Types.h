// Definition of data types used in L3STER

#ifndef L3STER_INCGUARD_TYPES_H
#define L3STER_INCGUARD_TYPES_H

#include <cstddef>

//  n   - node
//  el  - element
//  q   - quadrature
//  val - value
//
//  id  - ID
//  o   - order
//  dim - dimension

namespace lstr
{
    namespace types
    {
        using dim_t     = size_t;
        using n_id_t    = size_t;
        using el_id_t   = size_t;
        using el_o_t    = size_t;
        using el_dim_t  = size_t;
        using d_id_t    = size_t;
        using val_t     = double;
        using q_o_t     = size_t;
    }
}

#endif      // end include guard