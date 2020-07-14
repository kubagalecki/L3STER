#include "catch2/catch.hpp"
#include "l3ster.hpp"

#include <array>
#include <cmath>
#include <numeric>
#include <tuple>

constexpr double tol = 1e-10;

TEST_CASE("1D Gauss-Legendre quadrature, 1 point", "[quadrature]")
{
    const auto& ref_quad =
        lstr::quad::ReferenceQuadrature< lstr::quad::QuadratureTypes::GLeg, 0 >::value;

    REQUIRE(ref_quad.size == 1);
    REQUIRE(ref_quad.dim == 1);

    Approx qp_0 = Approx(0).margin(tol);
    Approx w_0  = Approx(2).margin(tol);

    CHECK(ref_quad.getQPoints()[0][0] == qp_0);
    CHECK(ref_quad.getWeights()[0] == w_0);
}

TEST_CASE("1D Gauss-Legendre quadrature, 2 point", "[quadrature]")
{
    const auto& ref_quad =
        lstr::quad::ReferenceQuadrature< lstr::quad::QuadratureTypes::GLeg, 2 >::value;

    REQUIRE(ref_quad.size == 2);
    REQUIRE(ref_quad.dim == 1);

    Approx qp_0 = Approx(-0.57735026919).margin(tol);
    Approx qp_1 = Approx(0.57735026919).margin(tol);
    Approx w    = Approx(1).margin(tol);

    CHECK(ref_quad.getQPoints()[0].front() == qp_0);
    CHECK(ref_quad.getQPoints()[1].front() == qp_1);
    CHECK(ref_quad.getWeights()[0] == w);
    CHECK(ref_quad.getWeights()[0] == w);
}

TEST_CASE("1D Gauss-Legendre quadrature, 3 point", "[quadrature]")
{
    const auto& ref_quad =
        lstr::quad::ReferenceQuadrature< lstr::quad::QuadratureTypes::GLeg, 4 >::value;

    REQUIRE(ref_quad.size == 3);
    REQUIRE(ref_quad.dim == 1);

    Approx qp_0 = Approx(-0.77459666924).margin(tol);
    Approx qp_1 = Approx(0).margin(tol);
    Approx qp_2 = Approx(0.77459666924).margin(tol);
    Approx w_0  = Approx(0.5555555556).margin(tol);
    Approx w_1  = Approx(0.8888888889).margin(tol);
    Approx w_2  = Approx(0.5555555556).margin(tol);

    CHECK(ref_quad.getQPoints()[0].front() == qp_0);
    CHECK(ref_quad.getQPoints()[1].front() == qp_1);
    CHECK(ref_quad.getQPoints()[2].front() == qp_2);
    CHECK(ref_quad.getWeights()[0] == w_0);
    CHECK(ref_quad.getWeights()[1] == w_1);
    CHECK(ref_quad.getWeights()[2] == w_2);
}

TEST_CASE("Gauss-Legendre quadratures for line element", "[quadrature]")
{
    const auto element = lstr::mesh::Element< lstr::mesh::ElementTypes::Line, 1 >{{1, 2}};

    SECTION("1 point quadrature")
    {
        const lstr::quad::QuadratureGenerator< lstr::quad::QuadratureTypes::GLeg, 1 > quad_gen;
        const auto& quadrature = quad_gen.get(element);

        REQUIRE(quadrature.size == 1);
        REQUIRE(quadrature.dim == 1);

        Approx qp_0 = Approx(0).margin(tol);
        Approx w_0  = Approx(2).margin(tol);

        CHECK(quadrature.getQPoints()[0][0] == qp_0);
        CHECK(quadrature.getWeights()[0] == w_0);
    }

    SECTION("2 point quadrature")
    {
        const lstr::quad::QuadratureGenerator< lstr::quad::QuadratureTypes::GLeg, 3 > quad_gen;
        const auto& quadrature = quad_gen.get(element);

        REQUIRE(quadrature.size == 2);
        REQUIRE(quadrature.dim == 1);

        Approx qp_0 = Approx(-0.57735026919).margin(tol);
        Approx qp_1 = Approx(0.57735026919).margin(tol);
        Approx w    = Approx(1).margin(tol);

        CHECK(quadrature.getQPoints()[0].front() == qp_0);
        CHECK(quadrature.getQPoints()[1].front() == qp_1);
        CHECK(quadrature.getWeights()[0] == w);
        CHECK(quadrature.getWeights()[0] == w);
    }

    SECTION("3 point quadrature")
    {
        const lstr::quad::QuadratureGenerator< lstr::quad::QuadratureTypes::GLeg, 5 > quad_gen;
        const auto& quadrature = quad_gen.get(element);

        REQUIRE(quadrature.size == 3);
        REQUIRE(quadrature.dim == 1);

        Approx qp_0 = Approx(-0.77459666924).margin(tol);
        Approx qp_1 = Approx(0).margin(tol);
        Approx qp_2 = Approx(0.77459666924).margin(tol);
        Approx w_0  = Approx(0.5555555556).margin(tol);
        Approx w_1  = Approx(0.8888888889).margin(tol);
        Approx w_2  = Approx(0.5555555556).margin(tol);

        CHECK(quadrature.getQPoints()[0].front() == qp_0);
        CHECK(quadrature.getQPoints()[1].front() == qp_1);
        CHECK(quadrature.getQPoints()[2].front() == qp_2);
        CHECK(quadrature.getWeights()[0] == w_0);
        CHECK(quadrature.getWeights()[1] == w_1);
        CHECK(quadrature.getWeights()[2] == w_2);
    }
}

TEST_CASE("Gauss-Legendre quadratures for quadrilateral element", "[quadrature]")
{
    const auto element = lstr::mesh::Element< lstr::mesh::ElementTypes::Quad, 1 >{{1, 2, 3, 4}};

    constexpr auto integrate_over_quad = [](const auto& quadrature, const auto& fun) {
        return std::transform_reduce(
            quadrature.getQPoints().cbegin(),
            quadrature.getQPoints().cend(),
            quadrature.getWeights().cbegin(),
            0.,
            std::plus<>{},
            [&fun](const std::array< lstr::types::val_t, 2 >& qp, const lstr::types::val_t& w) {
                return std::apply(fun, qp) * w;
            });
    };

    constexpr auto o0_fun = [](const auto&, const auto&) {
        return 1.;
    };
    Approx o0_int = Approx(4.).margin(tol);

    constexpr auto o1_fun = [](const auto& xi, const auto& eta) {
        return 2. * xi + 3. * eta + 1.;
    };
    Approx o1_int = Approx(4.).margin(tol);

    constexpr auto o2_fun = [](const auto& xi, const auto& eta) {
        return 2. * xi * xi + xi + 3. * eta * eta + 2. * eta + 1.;
    };
    Approx o2_int = Approx(10.66666666667).margin(tol);

    constexpr auto o3_fun = [](const auto& xi, const auto& eta) {
        return 3. * xi * xi * xi + 2. * xi * xi + xi + 4. * eta * eta * eta + 3. * eta * eta +
               2. * eta + 1.;
    };
    Approx o3_int = Approx(10.66666666667).margin(tol);

    constexpr auto o4_fun = [](const auto& xi, const auto& eta) {
        return 4. * xi * xi * xi * xi + 3. * xi * xi * xi + 2. * xi * xi + xi +
               5. * eta * eta * eta * eta + 4. * eta * eta * eta + 3. * eta * eta + 2. * eta + 1.;
    };
    Approx o4_int = Approx(17.86666666667).margin(tol);

    constexpr auto o5_fun = [](const auto& xi, const auto& eta) {
        return 5. * xi * xi * xi * xi * xi + 4. * xi * xi * xi * xi + 3. * xi * xi * xi +
               2. * xi * xi + xi + 6. * eta * eta * eta * eta * eta + 5. * eta * eta * eta * eta +
               4. * eta * eta * eta + 3. * eta * eta + 2. * eta + 1.;
    };
    Approx o5_int = Approx(17.86666666667).margin(tol);

    SECTION("1 point quadrature")
    {
        const lstr::quad::QuadratureGenerator< lstr::quad::QuadratureTypes::GLeg, 1 > quad_gen;
        const auto& quadrature = quad_gen.get(element);

        REQUIRE(quadrature.size == 1);
        REQUIRE(quadrature.dim == 2);

        Approx qp_0 = Approx(0).margin(tol);
        Approx w_0  = Approx(4).margin(tol);
        CHECK(quadrature.getQPoints()[0][0] == qp_0);
        CHECK(quadrature.getQPoints()[0][1] == qp_0);
        CHECK(quadrature.getWeights()[0] == w_0);
    }

    SECTION("4 point quadrature")
    {
        const lstr::quad::QuadratureGenerator< lstr::quad::QuadratureTypes::GLeg, 3 > quad_gen;
        const auto& quadrature = quad_gen.get(element);

        REQUIRE(quadrature.size == 4);
        REQUIRE(quadrature.dim == 2);

        CHECK(integrate_over_quad(quadrature, o0_fun) == o0_int);
        CHECK(integrate_over_quad(quadrature, o1_fun) == o1_int);
        CHECK(integrate_over_quad(quadrature, o2_fun) == o2_int);
        CHECK(integrate_over_quad(quadrature, o3_fun) == o3_int);
    }

    SECTION("9 point quadrature")
    {
        const lstr::quad::QuadratureGenerator< lstr::quad::QuadratureTypes::GLeg, 5 > quad_gen;
        const auto& quadrature = quad_gen.get(element);

        REQUIRE(quadrature.size == 9);
        REQUIRE(quadrature.dim == 2);

        CHECK(integrate_over_quad(quadrature, o0_fun) == o0_int);
        CHECK(integrate_over_quad(quadrature, o1_fun) == o1_int);
        CHECK(integrate_over_quad(quadrature, o2_fun) == o2_int);
        CHECK(integrate_over_quad(quadrature, o3_fun) == o3_int);
        CHECK(integrate_over_quad(quadrature, o4_fun) == o4_int);
        CHECK(integrate_over_quad(quadrature, o5_fun) == o5_int);
    }
}
