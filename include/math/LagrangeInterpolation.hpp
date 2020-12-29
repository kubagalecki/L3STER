#ifndef L3STER_MATH_LAGRANGEINTERPOLATION_HPP
#define L3STER_MATH_LAGRANGEINTERPOLATION_HPP

#include "math/Polynomial.hpp"

#include <algorithm>
#include <array>
#include <concepts>

namespace lstr::math
{
template < std::floating_point T, size_t N >
constexpr Polynomial< T, N - 1 > lagrangeInterp(const std::array< T, N >& x,
                                                const std::array< T, N >& y)
{
    static_assert(N > 1, "At least two points are needed for interpolation");

    // Algorithm: Sum N polynomials l_i, such that l_i has roots at all x except x[i], and l_i(x[i])
    // == y[i] Note: This method is accurate up until approximately N == 16
    // TODO: Come up with/look up more accurate algorithm
    // TODO: Explicitly test this function for basis functions of max order, as this is primarily
    // what this utility is meant for.

    std::array< T, N > lag_coefs{};
    for (size_t i = 0; i < N; ++i)
    {
        std::array< T, N - 1 > roots;
        std::copy(x.cbegin(), x.cbegin() + i, roots.begin());
        std::copy(x.cbegin() + i + 1, x.cend(), roots.begin() + i);
        std::array< T, N > l_i{};
        l_i.front() = 1.;
        for (size_t j = 0; j < N - 1; ++j)
        {
            for (size_t k = j + 1; k > 0; --k)
                l_i[k] -= l_i[k - 1] * roots[j];
        }
        const PolynomialView root_poly{l_i};
        const T              s = y[i] / root_poly.evaluate(x[i]);
        for (size_t k = 0; k < N; ++k)
            lag_coefs[k] += s * l_i[k];
    }
    return Polynomial{lag_coefs};
}
} // namespace lstr::math
#endif // L3STER_MATH_LAGRANGEINTERPOLATION_HPP
