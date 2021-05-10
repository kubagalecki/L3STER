#ifndef L3STER_MATH_LAGRANGEINTERPOLATION_HPP
#define L3STER_MATH_LAGRANGEINTERPOLATION_HPP

#include "math/Polynomial.hpp"

#include <algorithm>
#include <array>
#include <concepts>

namespace lstr
{
template < std::floating_point T, size_t N >
requires(N > 1) constexpr Polynomial< T, N - 1 > lagrangeInterp(const std::array< T, N >& x,
                                                                const std::array< T, N >& y)
{
    // Algorithm: Sum N polynomials l_i, such that l_i has roots at all x except x[i], and l_i(x[i]) == y[i]
    // Note: This method is accurate up until approximately N == 16
    // TODO: Come up with/look up more accurate algorithm
    // TODO: Explicitly test this function for basis functions of max order, as this is primarily what this utility is
    // meant for.

    std::array< T, N > lag_coefs{};
    for (size_t i = 0; i < N; ++i)
    {
        std::array< T, N - 1 > roots;
        std::copy_n(begin(x), i, roots.begin());
        std::ranges::copy(x | std::views::drop(i + 1), begin(roots) + i);
        std::array< T, N > l_i{};
        l_i.front() = 1.;
        for (auto j : std::views::iota(0u, N - 1))
        {
            for (size_t k = j + 1; k > 0; --k)
                l_i[k] -= l_i[k - 1] * roots[j];
        }
        const PolynomialView root_poly{l_i};
        const T              s = y[i] / root_poly.evaluate(x[i]);
        for (auto j : std::views::iota(0u, N))
            lag_coefs[j] += s * l_i[j];
    }
    return Polynomial{lag_coefs};
}
} // namespace lstr
#endif // L3STER_MATH_LAGRANGEINTERPOLATION_HPP
