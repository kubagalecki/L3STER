#ifndef L3STER_MATH_LAGRANGEINTERPOLATION_HPP
#define L3STER_MATH_LAGRANGEINTERPOLATION_HPP

#include "l3ster/math/Polynomial.hpp"

#include <algorithm>
#include <array>
#include <concepts>

namespace lstr::math
{
template < std::floating_point T, size_t N >
constexpr Polynomial< T, N - 1 > lagrangeInterp(const std::array< T, N >& x, const std::array< T, N >& y)
    requires(N > 1)
{
    // Algorithm: Sum N polynomials l_i, such that l_i has roots at all x except x[i], and l_i(x[i]) == y[i]
    // Note: This method is accurate up until approximately N == 16
    // TODO: Come up with/look up more accurate algorithm
    // TODO: Explicitly test this function for basis functions of max order, as this is primarily what this utility is
    // meant for.

    std::array< T, N > lag_coefs{};
    for (auto i : std::views::iota(0, ptrdiff_t{N}))
    {
        std::array< T, N - 1 > roots;
        std::copy_n(begin(x), i, roots.begin());
        std::ranges::copy(x | std::views::drop(i + 1), begin(roots) + i);
        std::array< T, N > l_i{};
        l_i.front() = 1.;
        for (auto j : std::views::iota(0, ptrdiff_t{N - 1}))
        {
            for (ptrdiff_t k = j + 1; k > 0; --k)
                l_i[k] -= l_i[k - 1] * roots[j];
        }
        const PolynomialView root_poly{l_i};
        const T              s = y[i] / root_poly.evaluate(x[i]);
        for (auto j : std::views::iota(0, ptrdiff_t{N}))
            lag_coefs[j] += s * l_i[j];
    }
    return Polynomial{lag_coefs};
}
} // namespace lstr::math
#endif // L3STER_MATH_LAGRANGEINTERPOLATION_HPP
