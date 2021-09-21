#ifndef L3STER_MATH_POLYNOMIAL_HPP
#define L3STER_MATH_POLYNOMIAL_HPP

#include "l3ster/util/Concepts.hpp"

#include "Eigen/Dense"

#include <algorithm>
#include <array>
#include <concepts>
#include <numeric>
#include <utility>

namespace lstr
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

template < array Arg >
PolynomialView(const Arg&) -> PolynomialView< typename Arg::value_type, std::tuple_size_v< Arg > - 1 >;

template < std::floating_point T, size_t ORDER >
struct Polynomial
{
    using value_type              = T;
    static constexpr size_t order = ORDER;

    [[nodiscard]] constexpr T                          evaluate(T x) const { return PolynomialView{coefs}.evaluate(x); }
    [[nodiscard]] constexpr Polynomial< T, ORDER + 1 > integral() const { return PolynomialView{coefs}.integral(); }
    [[nodiscard]] constexpr Polynomial< T, ORDER == 0 ? 0 : ORDER - 1 > derivative() const
    {
        return PolynomialView{coefs}.derivative();
    }
    [[nodiscard]] std::array< std::complex< T >, ORDER > roots() const { return PolynomialView{coefs}.roots(); }

    std::array< T, ORDER + 1u > coefs;
};

template < array Arg >
Polynomial(const Arg&) -> Polynomial< typename Arg::value_type, std::tuple_size_v< Arg > - 1 >;

template < std::floating_point T, size_t ORDER >
[[nodiscard]] constexpr T PolynomialView< T, ORDER >::evaluate(T x) const
{
    T ret = 0;
    for (const auto& c : coefs.get())
    {
        ret *= x;
        ret += c;
    }
    return ret;
}

template < std::floating_point T, size_t ORDER >
[[nodiscard]] constexpr Polynomial< T, ORDER + 1 > PolynomialView< T, ORDER >::integral() const
{
    std::array< T, ORDER + 1 > int_coefs;
    std::ranges::generate(int_coefs | std::views::reverse, [i = 1u]() mutable { return 1. / static_cast< T >(i++); });
    std::array< T, ORDER + 2 > ret;
    ret.back() = 0.; // Integration constant = 0
    std::ranges::transform(coefs.get(), int_coefs, ret.begin(), std::multiplies{});
    return Polynomial< T, ORDER + 1 >{ret};
}

template < std::floating_point T, size_t ORDER >
[[nodiscard]] constexpr Polynomial< T, ORDER == 0 ? 0 : ORDER - 1 > PolynomialView< T, ORDER >::derivative() const
{
    using ret_t = Polynomial< T, ORDER == 0 ? 0 : ORDER - 1 >;
    if constexpr (ORDER == 0)
        return ret_t{{0.}};
    else
    {
        std::array< T, ORDER > ret;
        std::ranges::transform(coefs.get() | std::views::take(ptrdiff_t{ORDER}),
                               std::views::iota(1u, ORDER + 1u) | std::views::reverse |
                                   std::views::transform([](size_t v) { return static_cast< T >(v); }),
                               begin(ret),
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
    const auto eig = comp_mat.eigenvalues();

    std::array< complex_t, ORDER > ret;
    std::ranges::copy(std::views::counted(eig.data(), eig.size()), ret.begin());
    std::ranges::sort(ret, [](complex_t a, complex_t b) { return a.real() < b.real(); });
    return ret;
}

template < std::floating_point T, size_t O1, size_t O2 >
constexpr Polynomial< T, O1 + O2 > operator*(const Polynomial< T, O1 >& a, const Polynomial< T, O2 >& b)
{
    Polynomial< T, O1 + O2 > ret{};
    ret.coefs.fill(0.);
    std::ranges::for_each(a, [&, a_ind = 0](T ac) mutable {
        std::ranges::for_each(b, [&, b_ind = 0u](T bc) mutable { ret.coefs[a_ind + b_ind++] += ac * bc; });
        ++a_ind;
    });
    return ret;
}

template < std::floating_point T, size_t O1, size_t O2 >
constexpr Polynomial< T, std::max(O1, O2) > operator+(const Polynomial< T, O1 >& a, const Polynomial< T, O2 >& b)
{
    constexpr auto poly_sum = [](const auto& poly_low, const auto& poly_high) {
        constexpr auto          O_low  = std::min(O1, O2);
        constexpr auto          O_high = std::max(O1, O2);
        constexpr size_t        O_diff = O_high - O_low;
        Polynomial< T, O_high > ret;
        std::ranges::copy(poly_high.coef | std::views::take(ptrdiff_t{O_diff}), ret.coef.begin());
        std::ranges::transform(poly_low.coef,
                               poly_high.coef | std::views::drop(ptrdiff_t{O_diff}),
                               ret.coef.begin() + O_diff,
                               std::plus{});
    };
    if constexpr (O1 < O2)
        return poly_sum(a, b);
    else
        return poly_sum(b, a);
}
} // namespace lstr
#endif // L3STER_MATH_POLYNOMIAL_HPP
