#include "catch2/catch.hpp"
#include "l3ster.h"

#include <cmath>

constexpr double tol = 1e-10;

TEST_CASE("1D Gauss-Legendre quadrature, 1 point", "[quadrature]")
{
    const auto& ref_quad = lstr::quad::ReferenceQuadrature<lstr::quad::QuadratureTypes::GLeg, 0>::value;

    REQUIRE(ref_quad.size == 1);
    REQUIRE(ref_quad.dim == 1);

    Approx qp_0 = Approx(0).margin(tol);
    Approx w_0 = Approx(2).margin(tol);
    CHECK(ref_quad.getQPoints()[0][0] == qp_0);
    CHECK(ref_quad.getWeights()[0] == w_0);
}

TEST_CASE("1D Gauss-Legendre quadrature, 2 point", "[quadrature]")
{
    const auto& ref_quad = lstr::quad::ReferenceQuadrature<lstr::quad::QuadratureTypes::GLeg, 2>::value;

    REQUIRE(ref_quad.size == 2);
    REQUIRE(ref_quad.dim == 1);

    Approx qp_0 = Approx(-0.57735026919).margin(tol);
    Approx qp_1 = Approx(0.57735026919).margin(tol);
    Approx w = Approx(1).margin(tol);
    CHECK(ref_quad.getQPoints()[0].front() == qp_0);
    CHECK(ref_quad.getQPoints()[1].front() == qp_1);
    CHECK(ref_quad.getWeights()[0] == w);
    CHECK(ref_quad.getWeights()[0] == w);
}

TEST_CASE("1D Gauss-Legendre quadrature, 3 point", "[quadrature]")
{
    const auto& ref_quad = lstr::quad::ReferenceQuadrature<lstr::quad::QuadratureTypes::GLeg, 4>::value;

    REQUIRE(ref_quad.size == 3);
    REQUIRE(ref_quad.dim == 1);

    Approx qp_0 = Approx(-0.77459666924).margin(tol);
    Approx qp_1 = Approx(0).margin(tol);
    Approx qp_2 = Approx(0.77459666924).margin(tol);
    Approx w_0 = Approx(0.5555555556).margin(tol);
    Approx w_1 = Approx(0.8888888889).margin(tol);
    Approx w_2 = Approx(0.5555555556).margin(tol);
    CHECK(ref_quad.getQPoints()[0].front() == qp_0);
    CHECK(ref_quad.getQPoints()[1].front() == qp_1);
    CHECK(ref_quad.getQPoints()[2].front() == qp_2);
    CHECK(ref_quad.getWeights()[0] == w_0);
    CHECK(ref_quad.getWeights()[1] == w_1);
    CHECK(ref_quad.getWeights()[2] == w_2);
}
