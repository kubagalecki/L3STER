#ifndef L3STER_MATH_LOBATTO_HPP
#define L3STER_MATH_LOBATTO_HPP

#include "l3ster/math/Legendre.hpp"

namespace lstr::math
{
template < std::floating_point T, size_t N >
constexpr auto getLobattoPolynomial()
{
    return getLegendrePolynomial< T, N + 1 >().derivative();
}
} // namespace lstr::math
#endif // L3STER_MATH_LOBATTO_HPP
