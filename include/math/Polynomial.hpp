#ifndef L3STER_MATH_POLYNOMIAL_HPP
#define L3STER_MATH_POLYNOMIAL_HPP

#include "util/Concepts.hpp"

#include "Eigen/Dense"

#include <algorithm>
#include <array>
#include <concepts>
#include <numeric>
#include <utility>

namespace lstr::math
{
template < std::floating_point T, size_t ORDER >
struct Polynomial;

template < std::floating_point T, size_t ORDER >
struct PolynomialView
{
    using value_type              = T;
    static constexpr size_t order = ORDER;

    [[nodiscard]] constexpr T                                           evaluate(T x) const;
    [[nodiscard]] constexpr Polynomial< T, ORDER + 1 >                  integral() const;
    [[nodiscard]] constexpr Polynomial< T, ORDER == 0 ? 0 : ORDER - 1 > derivative() const;
    [[nodiscard]] std::array< std::complex< T >, ORDER >                roots() const;

    std::reference_wrapper< const std::array< T, ORDER + 1u > > coefs;
};

template < util::array Arg >
PolynomialView(const Arg&)
    -> PolynomialView< typename Arg::value_type, std::tuple_size_v< Arg > - 1 >;

template < std::floating_point T, size_t ORDER >
struct Polynomial
{
    using value_type              = T;
    static constexpr size_t order = ORDER;

    [[nodiscard]] constexpr T evaluate(T x) const
    {
        return PolynomialView{coefs}.evaluate(x);
    }
    [[nodiscard]] constexpr Polynomial< T, ORDER + 1 > integral() const
    {
        return PolynomialView{coefs}.integral();
    }
    [[nodiscard]] constexpr Polynomial< T, ORDER == 0 ? 0 : ORDER - 1 > derivative() const
    {
        return PolynomialView{coefs}.derivative();
    }
    [[nodiscard]] std::array< std::complex< T >, ORDER > roots() const
    {
        return PolynomialView{coefs}.roots();
    }

    std::array< T, ORDER + 1u > coefs;
};

template < util::array Arg >
Polynomial(const Arg&) -> Polynomial< typename Arg::value_type, std::tuple_size_v< Arg > - 1 >;

template < std::floating_point T, size_t ORDER >
[[nodiscard]] constexpr T PolynomialView< T, ORDER >::evaluate(T x) const
{
    T ret         = coefs.get().back();
    T current_exp = x;
    std::for_each(coefs.get().crbegin() + 1, coefs.get().crend(), [&](T c) {
        ret += c * current_exp;
        current_exp *= x;
    });
    return ret;

    /*
    T y = 0.;
    std::for_each(coefs.get().crbegin(), coefs.get().crend(), [&, exponent = 0.](T a) mutable {
        y += a * pow(x, exponent++);
    });
    return y;
     */
}

template < std::floating_point T, size_t ORDER >
[[nodiscard]] constexpr Polynomial< T, ORDER + 1 > PolynomialView< T, ORDER >::integral() const
{
    std::array< T, ORDER + 1 > int_coefs;
    std::generate(int_coefs.rbegin(), int_coefs.rend(), [i = 1u]() mutable {
        return 1. / static_cast< T >(i++);
    });
    std::array< T, ORDER + 2 > ret;
    ret.back() = 0.; // Integration constant = 0
    std::transform(coefs.get().cbegin(),
                   coefs.get().cend(),
                   int_coefs.cbegin(),
                   ret.begin(),
                   std::multiplies{});
    return Polynomial< T, ORDER + 1 >{ret};
}

template < std::floating_point T, size_t ORDER >
[[nodiscard]] constexpr Polynomial< T, ORDER == 0 ? 0 : ORDER - 1 >
PolynomialView< T, ORDER >::derivative() const
{
    using ret_t = Polynomial< T, ORDER == 0 ? 0 : ORDER - 1 >;
    if constexpr (ORDER == 0)
        return ret_t{{0.}};
    else
    {
        std::array< T, ORDER > ret;
        auto                   der_coefs = ret;
        std::iota(der_coefs.rbegin(), der_coefs.rend(), 1);
        std::transform(coefs.get().cbegin(),
                       coefs.get().cend() - 1,
                       der_coefs.cbegin(),
                       ret.begin(),
                       std::multiplies{});
        return ret_t{ret};
    }
}

template < std::floating_point T, size_t ORDER >
[[nodiscard]] std::array< std::complex< T >, ORDER > PolynomialView< T, ORDER >::roots() const
{
    using complex_t = std::complex< T >;

    if constexpr (ORDER == 0)
        throw std::logic_error{"Cannot find roots of polynomial of order 0"};

    if constexpr (ORDER == 1)
        return std::array< complex_t, 1 >{complex_t{-coefs.get().back() / coefs.get().front(), 0.}};

    // Create and populate companion matrix
    Eigen::Matrix< T, ORDER, ORDER > comp_mat = Eigen::Matrix< T, ORDER, ORDER >::Zero();
    comp_mat(0, ORDER - 1)                    = -coefs.get().back() / coefs.get().front();
    for (size_t i = 0; i < ORDER - 1; ++i)
    {
        comp_mat(i + 1, i)         = 1.;
        comp_mat(i + 1, ORDER - 1) = -coefs.get()[ORDER - 1 - i] / coefs.get().front();
    }

    auto eig = comp_mat.eigenvalues();
    auto ret = std::array< complex_t, ORDER >{};
    std::copy(eig.begin(), eig.end(), ret.begin());
    std::sort(ret.begin(), ret.end(), [](complex_t a, complex_t b) { return a.real() < b.real(); });
    return ret;
}

template < std::floating_point T, size_t O1, size_t O2 >
constexpr Polynomial< T, O1 + O2 > operator*(const Polynomial< T, O1 >& a,
                                             const Polynomial< T, O2 >& b)
{
    Polynomial< T, O1 + O2 > ret{};
    ret.coefs.fill(0.);
    std::for_each(a.cbegin(), a.cend(), [&, a_ind = 0](const T& ac) mutable {
        std::for_each(b.coefs.cbegin(), b.coefs.cend(), [&, b_ind = 0u](const T& bc) mutable {
            ret.coefs[a_ind + b_ind++] += ac * bc;
        });
        ++a_ind;
    });
    return ret;
}

template < std::floating_point T, size_t O1, size_t O2 >
constexpr Polynomial< T, std::max(O1, O2) > operator+(const Polynomial< T, O1 >& a,
                                                      const Polynomial< T, O2 >& b)
{
    constexpr auto   O_low    = std::min(O1, O2);
    constexpr auto   O_high   = std::max(O1, O2);
    constexpr size_t O_diff   = O_high - O_low;
    constexpr auto   poly_sum = [&](const auto& poly_low, const auto& poly_high) {
        Polynomial< T, O_high > ret;
        std::copy(poly_high.coef.cbegin(), poly_high.coef.cbegin() + O_diff, ret.coef.begin());
        std::transform(poly_low.coef.cbegin(),
                       poly_low.coef.cend(),
                       poly_high.coef.cbegin() + O_diff,
                       ret.coef.begin() + O_diff,
                       std::plus{});
    };
    if constexpr (O1 < O2)
        return poly_sum(a, b);
    else
        return poly_sum(b, a);
}
} // namespace lstr::math
#endif // L3STER_MATH_POLYNOMIAL_HPP
