#ifndef L3STER_MATH_LOBATTORULEABSC_HPP
#define L3STER_MATH_LOBATTORULEABSC_HPP

#include "l3ster/math/Lobatto.hpp"

namespace lstr::math
{
namespace detail
{
template < std::floating_point T, size_t n_points >
auto computeLobattoRuleAbsc()
    requires(n_points > 1)
{
    if constexpr (n_points == 2)
        return std::array< T, 2 >{-1., 1.};
    else if constexpr (n_points == 3)
        return std::array< T, 3 >{-1., 0., 1.};
    else
    {
        std::array< T, n_points > retval;
        retval.front()           = -1;
        retval.back()            = 1.;
        const auto lobatto_roots = getLobattoPolynomial< T, n_points - 2 >().roots();
        std::ranges::transform(lobatto_roots, retval.begin() + 1, [](const std::complex< T >& c) { return c.real(); });
        return retval;
    }
}
} // namespace detail

template < std::floating_point T, size_t N >
const auto& getLobattoRuleAbsc()
{
    static const auto val = detail::computeLobattoRuleAbsc< T, N >();
    return val;
}
} // namespace lstr::math
#endif // L3STER_MATH_LOBATTORULEABSC_HPP
