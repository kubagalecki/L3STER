#include "catch2/catch.hpp"
#include "math/LagrangeInterpolation.hpp"
#include "math/Legendre.hpp"

#include <cmath>
#include <random>

using namespace lstr::math;

constexpr double tol = 1e-10;

TEST_CASE("Polynomial evaluation", "[math]")
{
    constexpr size_t     test_size      = 1u << 10;
    constexpr double     argument_range = 10.;
    constexpr std::array test_coefs{1., 2., 3.};
    constexpr auto       polynomial = Polynomial{test_coefs};

    std::mt19937                             prng{std::random_device{}()};
    std::uniform_real_distribution< double > dist{-argument_range, argument_range};
    const auto                               test_domain = [&] {
        std::array< double, test_size > ret{};
        std::generate(ret.begin(), ret.end(), [&] { return dist(prng); });
        return ret;
    }();

    // Note: using pow introduces significant error, so argument_range needs to be kept small
    const auto test_range = [&] {
        std::array< double, test_size > ret{};
        std::transform(test_domain.cbegin(), test_domain.cend(), ret.begin(), [&](double x) {
            double r = 0.;
            for (size_t i = 0; i < test_coefs.size(); ++i)
                r += test_coefs[i] * pow(x, static_cast< double >(test_coefs.size() - i - 1u));
            return r;
        });
        return ret;
    }();

    const auto computed_range = [&] {
        std::array< double, test_size > ret{};
        std::transform(test_domain.cbegin(), test_domain.cend(), ret.begin(), [&](double x) {
            return polynomial.evaluate(x);
        });
        return ret;
    }();

    constexpr auto approx = [](const double a, const double b) {
        return fabs(a - b) < tol;
    };

    CHECK(std::ranges::equal(test_range, computed_range, approx));
}

TEST_CASE("Polynomial integral", "[math]")
{
    constexpr auto cubic    = Polynomial{std::array{1., 2., 3.}};
    constexpr auto integral = cubic.integral();

    constexpr std::array expected_result{1. / 3., 1., 3., 0.};
    constexpr auto       expected_result_order = decltype(cubic)::order + 1u;

    REQUIRE(decltype(integral)::order == expected_result_order);
    for (size_t i = 0; i < expected_result_order; ++i)
    {
        CHECK(integral.coefs[i] == Approx(expected_result[i]).epsilon(tol));
    }
}

TEST_CASE("Polynomial derivative", "[math]")
{
    constexpr auto cubic      = Polynomial{std::array{1., 2., 3.}};
    constexpr auto derivative = cubic.derivative();

    constexpr std::array expected_result{2., 2.};
    constexpr auto       expected_result_order = decltype(cubic)::order - 1u;

    REQUIRE(decltype(derivative)::order == expected_result_order);
    for (size_t i = 0; i < expected_result_order; ++i)
    {
        CHECK(derivative.coefs[i] == Approx(expected_result[i]).epsilon(tol));
    }

    constexpr auto constant            = Polynomial{std::array{1.}};
    constexpr auto constant_derivative = constant.derivative();

    REQUIRE(decltype(constant_derivative)::order == 0);
    CHECK(constant_derivative.coefs[0] == 0.);
}

TEST_CASE("Polynomial roots", "[math]")
{
    CHECK(std::get< 0 >(Polynomial{std::array{-2., 1.}}.roots()).real() == Approx{.5}.epsilon(tol));

    const auto o2_roots = Polynomial{std::array{-1., 2., 3.}}.roots();
    CHECK(o2_roots[0].imag() == Approx{0.}.epsilon(tol));
    CHECK(o2_roots[1].imag() == Approx{0.}.epsilon(tol));
    CHECK(o2_roots[0].real() == Approx{-1.}.epsilon(tol));
    CHECK(o2_roots[1].real() == Approx{3.}.epsilon(tol));

    const auto o3_roots = Polynomial{std::array{-1., 2., 1., -2.}}.roots();
    CHECK(o3_roots[0].imag() == Approx{0.}.epsilon(tol));
    CHECK(o3_roots[1].imag() == Approx{0.}.epsilon(tol));
    CHECK(o3_roots[2].imag() == Approx{0.}.epsilon(tol));
    CHECK(o3_roots[0].real() == Approx{-1.}.epsilon(tol));
    CHECK(o3_roots[1].real() == Approx{1.}.epsilon(tol));
    CHECK(o3_roots[2].real() == Approx{2.}.epsilon(tol));
}

TEST_CASE("Lagrange interpolation", "[math]")
{
    constexpr size_t test_size      = 1u << 4;
    constexpr double argument_range = 1.;
    std::mt19937     prng{std::random_device{}()};
    const auto       make_random_array_spread = [&] {
        std::array< double, test_size > ret{};
        std::generate(ret.begin(), ret.end(), [&, x0 = -argument_range]() mutable {
            x0 += argument_range;
            return std::uniform_real_distribution< double >{x0, x0 + argument_range}(prng);
        });
        return ret;
    };
    const auto make_random_array = [&] {
        std::array< double, test_size > ret{};
        std::generate(ret.begin(), ret.end(), [&] {
            return std::uniform_real_distribution< double >{argument_range,
                                                            2 * argument_range}(prng);
        });
        return ret;
    };

    const auto x          = make_random_array_spread();
    const auto y          = make_random_array();
    const auto lag_poly   = lagrangeInterp(x, y);
    const auto y_computed = [&] {
        std::array< double, test_size > ret{};
        std::transform(
            x.cbegin(), x.cend(), ret.begin(), [&](double in) { return lag_poly.evaluate(in); });
        return ret;
    }();
    constexpr auto approx = [](const double a, const double b) {
        constexpr double lagrange_tol = 2e-3;
        return fabs(a - b) < lagrange_tol;
    };
    CHECK(std::ranges::equal(y, y_computed, approx));
}

TEST_CASE("Legendre polynomials", "[math]")
{
    constexpr auto approx = [](const double a, const double b) {
        return fabs(a - b) < tol;
    };
    constexpr auto lp2          = getLegendrePolynomial< double, 2 >();
    constexpr auto lp2_expected = Polynomial{std::array{1.5, 0., -.5}};
    CHECK(std::ranges::equal(lp2.coefs, lp2_expected.coefs, approx));

    constexpr auto lp3          = getLegendrePolynomial< double, 3 >();
    constexpr auto lp3_expected = Polynomial{std::array{2.5, 0., -1.5, 0.}};
    CHECK(std::ranges::equal(lp3.coefs, lp3_expected.coefs, approx));

    constexpr auto lp4          = getLegendrePolynomial< double, 4 >();
    constexpr auto lp4_expected = Polynomial{std::array{4.375, 0., -3.75, 0., .375}};
    CHECK(std::ranges::equal(lp4.coefs, lp4_expected.coefs, approx));
}
