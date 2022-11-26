#ifndef L3STER_QUAD_QUADRATURE_HPP
#define L3STER_QUAD_QUADRATURE_HPP

#include "l3ster/defs/Typedefs.h"

#include <array>

namespace lstr
{
template < q_l_t q_size, dim_t q_dim >
struct Quadrature
{
    using q_points_t = std::array< std::array< val_t, q_dim >, q_size >;
    using weights_t  = std::array< val_t, q_size >;

    static constexpr q_l_t size = q_size;
    static constexpr dim_t dim  = q_dim;

    q_points_t points;
    weights_t  weights;
};
template < size_t dim, size_t len >
Quadrature(std::array< std::array< val_t, dim >, len >, std::array< val_t, len >) -> Quadrature< len, dim >;
} // namespace lstr
#endif // L3STER_QUAD_QUADRATURE_HPP
