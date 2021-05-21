#ifndef L3STER_MESH_LOBATTO_HPP
#define L3STER_MESH_LOBATTO_HPP

#include "math/Legendre.hpp"

namespace lstr
{
template < std::floating_point T, size_t N >
constexpr auto getLobattoPolynomial()
{
    return getLegendrePolynomial< T, N + 1 >().derivative();
}
} // namespace lstr
#endif // L3STER_MESH_LOBATTO_HPP
