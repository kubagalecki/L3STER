#ifndef L3STER_MATH_POLYNOMIAL_HPP
#define L3STER_MATH_POLYNOMIAL_HPP

#include "l3ster/util/Concepts.hpp"
#include "l3ster/util/EigenUtils.hpp"
#include "l3ster/util/Ranges.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <numeric>
#include <utility>

namespace lstr::math
{
template < std::floating_point T, size_t ORDER >
struct Polynomial
{
    using value_type              = T;
    static constexpr size_t order = ORDER;

    [[nodiscard]] constexpr auto evaluate(T x) const -> T;
    [[nodiscard]] constexpr auto integral() const -> Polynomial< T, ORDER + 1 >;
    [[nodiscard]] constexpr auto derivative() const -> Polynomial< T, ORDER == 0 ? 0 : ORDER - 1 >;
    [[nodiscard]] auto           roots() const -> std::array< std::complex< T >, ORDER >;

    std::array< T, ORDER + 1u > coefs;
};

template < Array_c Arg >
Polynomial(const Arg&) -> Polynomial< typename Arg::value_type, std::tuple_size_v< Arg > - 1 >;

template < std::floating_point T, size_t ORDER >
[[nodiscard]] constexpr auto Polynomial< T, ORDER >::evaluate(T x) const -> T
{
    constexpr auto bit_width        = std::bit_width(ORDER);
    const auto     powers_2n        = std::invoke([&] {
        auto retval = std::array< T, bit_width >{};
        std::ranges::generate(retval, [x] mutable { return std::exchange(x, x * x); });
        return retval;
    });
    const auto     compute_x_to_ith = [&](size_t i) {
        return std::ranges::fold_left(
            std::views::zip_transform([](bool bit, T pow2) { return bit ? pow2 : T{1.}; }, util::bitView(i), powers_2n),
            T{1.},
            std::multiplies{});
    };
    return std::ranges::fold_left(
        std::views::zip_transform(std::multiplies{},
                                  std::views::iota(0uz, ORDER + 1) | std::views::transform(compute_x_to_ith),
                                  coefs | std::views::reverse),
        T{0.},
        std::plus{});
}

template < std::floating_point T, size_t ORDER >
[[nodiscard]] constexpr auto Polynomial< T, ORDER >::integral() const -> Polynomial< T, ORDER + 1 >
{
    std::array< T, ORDER + 1 > int_coefs;
    std::ranges::generate(int_coefs | std::views::reverse, [i = 1u] mutable { return 1. / static_cast< T >(i++); });
    auto retval         = Polynomial< T, ORDER + 1 >{};
    retval.coefs.back() = 0.; // Integration constant = 0
    std::ranges::transform(coefs, int_coefs, retval.coefs.begin(), std::multiplies{});
    return retval;
}

template < std::floating_point T, size_t ORDER >
[[nodiscard]] constexpr auto Polynomial< T, ORDER >::derivative() const -> Polynomial< T, ORDER == 0 ? 0 : ORDER - 1 >
{
    using ret_t = Polynomial< T, ORDER == 0 ? 0 : ORDER - 1 >;
    if constexpr (ORDER == 0)
        return ret_t{{0.}};
    else
    {
        auto retval = ret_t{};
        std::ranges::transform(coefs | std::views::take(ORDER),
                               std::views::iota(1uz, ORDER + 1) | std::views::reverse |
                                   std::views::transform([](size_t v) { return static_cast< T >(v); }),
                               retval.coefs.begin(),
                               std::multiplies{});
        return retval;
    }
}

template < std::floating_point T, size_t ORDER >
[[nodiscard]] auto Polynomial< T, ORDER >::roots() const -> std::array< std::complex< T >, ORDER >
{
    using complex_t = std::complex< T >;

    if constexpr (ORDER == 0)
        throw std::logic_error{"Cannot find roots of polynomial of order 0"};

    if constexpr (ORDER == 1)
        return std::array< complex_t, 1 >{complex_t{-coefs.back() / coefs.front(), 0.}};

    // Create and populate companion matrix
    Eigen::Matrix< T, ORDER, ORDER > comp_mat = Eigen::Matrix< T, ORDER, ORDER >::Zero();
    comp_mat(0, ORDER - 1)                    = -coefs.back() / coefs.front();
    for (size_t i = 0; i < ORDER - 1; ++i)
    {
        comp_mat(i + 1, i)         = 1.;
        comp_mat(i + 1, ORDER - 1) = -coefs[ORDER - 1 - i] / coefs.front();
    }
    const auto eig = comp_mat.eigenvalues();

    std::array< complex_t, ORDER > ret;
    std::ranges::copy(std::views::counted(eig.data(), eig.size()), ret.begin());
    std::ranges::sort(ret, [](complex_t a, complex_t b) { return a.real() < b.real(); });
    return ret;
}
} // namespace lstr::math
#endif // L3STER_MATH_POLYNOMIAL_HPP
