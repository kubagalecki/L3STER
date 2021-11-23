#include "l3ster/local_assembly/ComputeRefBasesAtQpoints.hpp"
#include "l3ster/quad/EvalQuadrature.hpp"

#include "catch2/catch.hpp"

static constexpr double tol = 1e-10;

using namespace lstr;

TEST_CASE("1D Gauss-Legendre quadrature, 1 point", "[quadrature]")
{
    const auto& ref_quad = getReferenceQuadrature< QuadratureTypes::GLeg, 0 >();

    REQUIRE(ref_quad.size == 1);
    REQUIRE(ref_quad.dim == 1);

    Approx qp_0 = Approx(0).margin(tol);
    Approx w_0  = Approx(2).margin(tol);

    CHECK(ref_quad.getPoints()[0][0] == qp_0);
    CHECK(ref_quad.getWeights()[0] == w_0);
}

TEST_CASE("1D Gauss-Legendre quadrature, 2 point", "[quadrature]")
{
    const auto& ref_quad = getReferenceQuadrature< QuadratureTypes::GLeg, 2 >();

    REQUIRE(ref_quad.size == 2);
    REQUIRE(ref_quad.dim == 1);

    Approx qp_0 = Approx(-0.57735026919).margin(tol);
    Approx qp_1 = Approx(0.57735026919).margin(tol);
    Approx w    = Approx(1).margin(tol);

    CHECK(ref_quad.getPoints()[0].front() == qp_0);
    CHECK(ref_quad.getPoints()[1].front() == qp_1);
    CHECK(ref_quad.getWeights()[0] == w);
    CHECK(ref_quad.getWeights()[0] == w);
}

TEST_CASE("1D Gauss-Legendre quadrature, 3 point", "[quadrature]")
{
    const auto& ref_quad = getReferenceQuadrature< QuadratureTypes::GLeg, 4 >();

    REQUIRE(ref_quad.size == 3);
    REQUIRE(ref_quad.dim == 1);

    Approx qp_0 = Approx(-0.77459666924).margin(tol);
    Approx qp_1 = Approx(0).margin(tol);
    Approx qp_2 = Approx(0.77459666924).margin(tol);
    Approx w_0  = Approx(0.55555555556).margin(tol);
    Approx w_1  = Approx(0.88888888889).margin(tol);
    Approx w_2  = Approx(0.55555555556).margin(tol);

    CHECK(ref_quad.getPoints()[0].front() == qp_0);
    CHECK(ref_quad.getPoints()[1].front() == qp_1);
    CHECK(ref_quad.getPoints()[2].front() == qp_2);
    CHECK(ref_quad.getWeights()[0] == w_0);
    CHECK(ref_quad.getWeights()[1] == w_1);
    CHECK(ref_quad.getWeights()[2] == w_2);
}

TEST_CASE("Gauss-Legendre quadratures for line element", "[quadrature]")
{
    SECTION("1 point quadrature")
    {
        const auto& quadrature = getQuadrature< QuadratureTypes::GLeg, 1, ElementTypes::Line >();

        REQUIRE(quadrature.size == 1);
        REQUIRE(quadrature.dim == 1);

        Approx qp_0 = Approx(0).margin(tol);
        Approx w_0  = Approx(2).margin(tol);

        CHECK(quadrature.getPoints()[0][0] == qp_0);
        CHECK(quadrature.getWeights()[0] == w_0);
    }

    SECTION("2 point quadrature")
    {
        const auto& quadrature = getQuadrature< QuadratureTypes::GLeg, 3, ElementTypes::Line >();

        REQUIRE(quadrature.size == 2);
        REQUIRE(quadrature.dim == 1);

        Approx qp_0 = Approx(-0.57735026919).margin(tol);
        Approx qp_1 = Approx(0.57735026919).margin(tol);
        Approx w    = Approx(1).margin(tol);

        CHECK(quadrature.getPoints()[0].front() == qp_0);
        CHECK(quadrature.getPoints()[1].front() == qp_1);
        CHECK(quadrature.getWeights()[0] == w);
        CHECK(quadrature.getWeights()[0] == w);
    }

    SECTION("3 point quadrature")
    {
        const auto& quadrature = getQuadrature< QuadratureTypes::GLeg, 5, ElementTypes::Line >();

        REQUIRE(quadrature.size == 3);
        REQUIRE(quadrature.dim == 1);

        Approx qp_0 = Approx(-0.77459666924).margin(tol);
        Approx qp_1 = Approx(0).margin(tol);
        Approx qp_2 = Approx(0.77459666924).margin(tol);
        Approx w_0  = Approx(0.55555555556).margin(tol);
        Approx w_1  = Approx(0.88888888889).margin(tol);
        Approx w_2  = Approx(0.55555555556).margin(tol);

        CHECK(quadrature.getPoints()[0].front() == qp_0);
        CHECK(quadrature.getPoints()[1].front() == qp_1);
        CHECK(quadrature.getPoints()[2].front() == qp_2);
        CHECK(quadrature.getWeights()[0] == w_0);
        CHECK(quadrature.getWeights()[1] == w_1);
        CHECK(quadrature.getWeights()[2] == w_2);
    }
}

static constexpr auto wrapQuadEvaluator(const auto& eval)
{
    return [&](ptrdiff_t, const auto& arr) noexcept {
        return std::apply(eval, arr);
    };
}

static inline constexpr auto zero_gen = []() noexcept {
    return 0.;
};

TEST_CASE("Gauss-Legendre quadratures for quadrilateral element", "[quadrature]")
{
    constexpr auto o0_fun = [](double, double) {
        return 1.;
    };
    Approx o0_int = Approx(4.).margin(tol);

    constexpr auto o1_fun = [](double xi, double eta) {
        return 2. * xi + 3. * eta + 1.;
    };
    Approx o1_int = Approx(4.).margin(tol);

    constexpr auto o2_fun = [](double xi, double eta) {
        return 2. * xi * xi + xi + 3. * eta * eta + 2. * eta + 1.;
    };
    Approx o2_int = Approx(10.666666666667).margin(tol);

    constexpr auto o3_fun = [](double xi, double eta) {
        return 3. * xi * xi * xi + 2. * xi * xi + xi + 4. * eta * eta * eta + 3. * eta * eta + 2. * eta + 1.;
    };
    Approx o3_int = Approx(10.666666666667).margin(tol);

    constexpr auto o4_fun = [](double xi, double eta) {
        return 4. * xi * xi * xi * xi + 3. * xi * xi * xi + 2. * xi * xi + xi + 5. * eta * eta * eta * eta +
               4. * eta * eta * eta + 3. * eta * eta + 2. * eta + 1.;
    };
    Approx o4_int = Approx(17.866666666667).margin(tol);

    constexpr auto o5_fun = [](double xi, double eta) {
        return 5. * xi * xi * xi * xi * xi + 4. * xi * xi * xi * xi + 3. * xi * xi * xi + 2. * xi * xi + xi +
               6. * eta * eta * eta * eta * eta + 5. * eta * eta * eta * eta + 4. * eta * eta * eta + 3. * eta * eta +
               2. * eta + 1.;
    };
    Approx o5_int = Approx(17.866666666667).margin(tol);

    SECTION("1 point quadrature")
    {
        const auto& quadrature = getQuadrature< QuadratureTypes::GLeg, 1, ElementTypes::Quad >();

        REQUIRE(quadrature.size == 1);
        REQUIRE(quadrature.dim == 2);

        Approx qp_0 = Approx(0).margin(tol);
        Approx w_0  = Approx(4).margin(tol);
        CHECK(quadrature.getPoints()[0][0] == qp_0);
        CHECK(quadrature.getPoints()[0][1] == qp_0);
        CHECK(quadrature.getWeights()[0] == w_0);
    }

    SECTION("4 point quadrature")
    {
        const auto& quadrature = getQuadrature< QuadratureTypes::GLeg, 3, ElementTypes::Quad >();

        REQUIRE(quadrature.size == 4);
        REQUIRE(quadrature.dim == 2);

        CHECK(evalQuadrature(o0_fun, quadrature) == o0_int);
        CHECK(evalQuadrature(o1_fun, quadrature) == o1_int);
        CHECK(evalQuadrature(o2_fun, quadrature) == o2_int);
        CHECK(evalQuadrature(o3_fun, quadrature) == o3_int);

        CHECK(evalQuadrature(wrapQuadEvaluator(o0_fun), quadrature, zero_gen) == o0_int);
        CHECK(evalQuadrature(wrapQuadEvaluator(o1_fun), quadrature, zero_gen) == o1_int);
        CHECK(evalQuadrature(wrapQuadEvaluator(o2_fun), quadrature, zero_gen) == o2_int);
        CHECK(evalQuadrature(wrapQuadEvaluator(o3_fun), quadrature, zero_gen) == o3_int);
    }

    SECTION("9 point quadrature")
    {
        const auto& quadrature = getQuadrature< QuadratureTypes::GLeg, 5, ElementTypes::Quad >();

        REQUIRE(quadrature.size == 9);
        REQUIRE(quadrature.dim == 2);

        CHECK(evalQuadrature(o0_fun, quadrature) == o0_int);
        CHECK(evalQuadrature(o1_fun, quadrature) == o1_int);
        CHECK(evalQuadrature(o2_fun, quadrature) == o2_int);
        CHECK(evalQuadrature(o3_fun, quadrature) == o3_int);
        CHECK(evalQuadrature(o4_fun, quadrature) == o4_int);
        CHECK(evalQuadrature(o5_fun, quadrature) == o5_int);

        CHECK(evalQuadrature(wrapQuadEvaluator(o0_fun), quadrature, zero_gen) == o0_int);
        CHECK(evalQuadrature(wrapQuadEvaluator(o1_fun), quadrature, zero_gen) == o1_int);
        CHECK(evalQuadrature(wrapQuadEvaluator(o2_fun), quadrature, zero_gen) == o2_int);
        CHECK(evalQuadrature(wrapQuadEvaluator(o3_fun), quadrature, zero_gen) == o3_int);
        CHECK(evalQuadrature(wrapQuadEvaluator(o4_fun), quadrature, zero_gen) == o4_int);
        CHECK(evalQuadrature(wrapQuadEvaluator(o5_fun), quadrature, zero_gen) == o5_int);
    }
}

TEST_CASE("Gauss-Legendre quadratures for hexahedral element", "[quadrature]")
{
    constexpr auto o0_fun = [](double, double, double) {
        return 1.;
    };
    Approx o0_int = Approx(8.).margin(tol);

    constexpr auto o1_fun = [](double x, double y, double z) {
        return x * y * z + x * y + y * z - z * x + x + y + z + 1.;
    };
    Approx o1_int = Approx(8.).margin(tol);

    constexpr auto o2_fun = [](double x, double y, double z) {
        return x * x * y * (y + 2.) * z * (z + 1.) + x * (y - 1.) + z * y * (y - 2.);
    };
    Approx o2_int = Approx(8. / 27.).margin(tol);

    constexpr auto o3_fun = [](double x, double y, double z) {
        return z * z * (z + 1.) * (x * x + x) + (y + 1.) * y * y;
    };
    Approx o3_int = Approx(32. / 9.).margin(tol);

    constexpr auto trig_fun = [](double x, double y, double z) {
        return sin(x) * tan(x) + sin(y) * cos(z) * cos(z);
    };
    Approx trig_int{-8. * (sin(1) - 2. * atanh(tan(.5)))};

    SECTION("1 point quadrature")
    {
        const auto& quadrature = getQuadrature< QuadratureTypes::GLeg, 1, ElementTypes::Hex >();

        REQUIRE(quadrature.size == 1);
        REQUIRE(quadrature.dim == 3);

        Approx qp_0 = Approx(0).margin(tol);
        Approx w_0  = Approx(8).margin(tol);
        CHECK(quadrature.getPoints()[0][0] == qp_0);
        CHECK(quadrature.getPoints()[0][1] == qp_0);
        CHECK(quadrature.getPoints()[0][2] == qp_0);
        CHECK(quadrature.getWeights()[0] == w_0);
    }

    SECTION("8 point quadrature")
    {
        const auto& quadrature = getQuadrature< QuadratureTypes::GLeg, 3, ElementTypes::Hex >();

        REQUIRE(quadrature.size == 8);
        REQUIRE(quadrature.dim == 3);

        CHECK(evalQuadrature(o0_fun, quadrature) == o0_int);
        CHECK(evalQuadrature(o1_fun, quadrature) == o1_int);
        CHECK(evalQuadrature(o2_fun, quadrature) == o2_int);
        CHECK(evalQuadrature(o3_fun, quadrature) == o3_int);

        CHECK(evalQuadrature(wrapQuadEvaluator(o0_fun), quadrature, zero_gen) == o0_int);
        CHECK(evalQuadrature(wrapQuadEvaluator(o1_fun), quadrature, zero_gen) == o1_int);
        CHECK(evalQuadrature(wrapQuadEvaluator(o2_fun), quadrature, zero_gen) == o2_int);
        CHECK(evalQuadrature(wrapQuadEvaluator(o3_fun), quadrature, zero_gen) == o3_int);
    }

    SECTION("512 point quadrature")
    {
        const auto& quadrature = getQuadrature< QuadratureTypes::GLeg, 15, ElementTypes::Hex >();

        REQUIRE(quadrature.size == 512);
        REQUIRE(quadrature.dim == 3);

        CHECK(evalQuadrature(trig_fun, quadrature) == trig_int);
        CHECK(evalQuadrature(wrapQuadEvaluator(trig_fun), quadrature, zero_gen) == trig_int);
    }
}
