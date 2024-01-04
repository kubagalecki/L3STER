#ifndef L3STER_MATH_LEGENDRE_HPP
#define L3STER_MATH_LEGENDRE_HPP

#include "l3ster/math/Polynomial.hpp"

namespace lstr::math
{
template < std::floating_point T, size_t N >
constexpr Polynomial< T, N > getLegendrePolynomial()
{
    using coef_array_t = std::array< T, N + 1 >;
    coef_array_t coefs{};
    if constexpr (N == 0)
        coefs[0] = 1.;
    else if constexpr (N == 1)
    {
        coefs[0] = 1.;
        coefs[1] = 0.;
    }
    else
    {
        coef_array_t P_n1{};
        coef_array_t P_n2{};
        *(P_n1.rbegin() + 1)  = 1.;
        coefs.back()          = -.5;
        *(coefs.rbegin() + 2) = 1.5;

        constexpr auto a = [](size_t x) {
            return static_cast< T >(2u * x - 1u) / static_cast< T >(x);
        };
        constexpr auto c = [](size_t x) {
            return static_cast< T >(x - 1u) / static_cast< T >(x);
        };

        // Compute L_{N} based on the recurrence relation L_{n} = x*a(n)*L_{n-1} - c*L_{n-2}
        for (size_t i = 3; i <= N; ++i)
        {
            const size_t index = N - i;
            std::copy(P_n1.cbegin() + index, P_n1.cend(), P_n2.begin() + index);
            std::copy(coefs.cbegin() + index, coefs.cend(), P_n1.begin() + index);
            std::transform(
                P_n1.cbegin() + index + 1u, P_n1.cend(), P_n2.cbegin() + index, coefs.begin() + index, [&](T c1, T c2) {
                    return a(i) * c1 - c(i) * c2;
                });
            coefs.back() = -c(i) * P_n2.back();
        }
    }
    return Polynomial{coefs};
}
} // namespace lstr::math

#endif // L3STER_MATH_LEGENDRE_HPP
