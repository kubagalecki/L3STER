#ifndef L3STER_MATH_COMPUTELOBATTORULEABSC_HPP
#define L3STER_MATH_COMPUTELOBATTORULEABSC_HPP

#include "math/Lobatto.hpp"

namespace lstr
{
template < std::floating_point T, size_t NPOINTS >
requires(NPOINTS > 1) auto computeLobattoRuleAbsc()
{
    if constexpr (NPOINTS == 2)
        return std::array< T, 2 >{-1., 1.};
    else if constexpr (NPOINTS == 3)
        return std::array< T, 3 >{-1., 0., 1.};
    else
    {
        std::array< T, NPOINTS > retval;
        retval.front()           = -1;
        retval.back()            = 1.;
        const auto lobatto_roots = getLobattoPolynomial< T, NPOINTS - 2 >().roots();
        std::ranges::transform(lobatto_roots, retval.begin() + 1, [](const std::complex< T >& c) { return c.real(); });
        return retval;
    }
}
} // namespace lstr
#endif // L3STER_MATH_COMPUTELOBATTORULEABSC_HPP
