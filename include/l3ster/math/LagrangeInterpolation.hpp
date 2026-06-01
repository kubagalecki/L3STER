#ifndef L3STER_MATH_LAGRANGEINTERPOLATION_HPP
#define L3STER_MATH_LAGRANGEINTERPOLATION_HPP

#include "l3ster/math/Polynomial.hpp"

#include <algorithm>
#include <array>
#include <concepts>

namespace lstr::math
{
template < std::floating_point T, size_t N >
constexpr auto lagrangeInterp(const std::array< T, N >& x, const std::array< T, N >& y) -> Polynomial< T, N - 1 >
    requires(N > 1)
{
    // Algorithm: Sum N polynomials l_i, such that l_i has roots at all x except x[i], and l_i(x[i]) == y[i]
    // Note: This method is accurate up to approximately N == 16
    // TODO: Come up with/look up more accurate algorithm

    auto retval = Polynomial< T, N - 1 >{};
    for (size_t i = 0; i != N; ++i)
    {
        auto       roots      = std::array< T, N - 1 >{};
        const auto write_iter = std::ranges::copy(x | std::views::take(i), roots.begin()).out;
        std::ranges::copy(x | std::views::drop(i + 1), write_iter);
        const auto l_i = std::invoke([&] {
            auto ret          = Polynomial< T, N - 1 >{};
            ret.coefs.front() = 1.;
            for (size_t j = 0; j != N - 1; ++j)
                for (size_t k = j + 1; k > 0; --k)
                    ret.coefs[k] -= ret.coefs[k - 1] * roots[j];
            return ret;
        });
        const auto s   = T{y[i] / l_i.evaluate(x[i])};
        for (size_t j = 0; j != N; ++j)
            retval.coefs[j] += s * l_i.coefs[j];
    }
    return retval;
}
} // namespace lstr::math
#endif // L3STER_MATH_LAGRANGEINTERPOLATION_HPP
