#include "l3ster.hpp"
#include "quad/InvokeQuadrature.hpp"

#include "catch2/catch.hpp"

static constexpr double tol = 1e-10;

TEST_CASE("1D Gauss-Legendre quadrature, 1 point", "[quadrature]")
{
    const auto& ref_quad = lstr::ReferenceQuadrature< lstr::QuadratureTypes::GLeg, 0 >::value;

    REQUIRE(ref_quad.size == 1);
    REQUIRE(ref_quad.dim == 1);

    Approx qp_0 = Approx(0).margin(tol);
    Approx w_0  = Approx(2).margin(tol);

    CHECK(ref_quad.getQPoints()[0][0] == qp_0);
    CHECK(ref_quad.getWeights()[0] == w_0);
}

TEST_CASE("1D Gauss-Legendre quadrature, 2 point", "[quadrature]")
{
    const auto& ref_quad = lstr::ReferenceQuadrature< lstr::QuadratureTypes::GLeg, 2 >::value;

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
    const auto& ref_quad = lstr::ReferenceQuadrature< lstr::QuadratureTypes::GLeg, 4 >::value;

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
    const lstr::ElementData< lstr::ElementTypes::Line, 1 > data{{lstr::Point{0., 0., 0.}, lstr::Point{0., 0., 0.}}};
    const auto element = lstr::Element< lstr::ElementTypes::Line, 1 >{{1, 2}, data, 0};

    SECTION("1 point quadrature")
    {
        const auto& quadrature = lstr::QuadratureGenerator< lstr::QuadratureTypes::GLeg, 1 >{}.get(element);

        REQUIRE(quadrature.size == 1);
        REQUIRE(quadrature.dim == 1);

        Approx qp_0 = Approx(0).margin(tol);
        Approx w_0  = Approx(2).margin(tol);

        CHECK(quadrature.getQPoints()[0][0] == qp_0);
        CHECK(quadrature.getWeights()[0] == w_0);
    }

    SECTION("2 point quadrature")
    {
        const auto& quadrature = lstr::QuadratureGenerator< lstr::QuadratureTypes::GLeg, 3 >{}.get(element);

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
        const auto& quadrature = lstr::QuadratureGenerator< lstr::QuadratureTypes::GLeg, 5 >{}.get(element);

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
    const lstr::ElementData< lstr::ElementTypes::Quad, 1 > data{
        {lstr::Point{0., 0., 0.}, lstr::Point{0., 0., 0.}, lstr::Point{0., 0., 0.}, lstr::Point{0., 0., 0.}}};
    const auto element = lstr::Element< lstr::ElementTypes::Quad, 1 >{{1, 2, 3, 4}, data, 0};

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
    Approx o2_int = Approx(10.66666666667).margin(tol);

    constexpr auto o3_fun = [](double xi, double eta) {
        return 3. * xi * xi * xi + 2. * xi * xi + xi + 4. * eta * eta * eta + 3. * eta * eta + 2. * eta + 1.;
    };
    Approx o3_int = Approx(10.66666666667).margin(tol);

    constexpr auto o4_fun = [](double xi, double eta) {
        return 4. * xi * xi * xi * xi + 3. * xi * xi * xi + 2. * xi * xi + xi + 5. * eta * eta * eta * eta +
               4. * eta * eta * eta + 3. * eta * eta + 2. * eta + 1.;
    };
    Approx o4_int = Approx(17.86666666667).margin(tol);

    constexpr auto o5_fun = [](double xi, double eta) {
        return 5. * xi * xi * xi * xi * xi + 4. * xi * xi * xi * xi + 3. * xi * xi * xi + 2. * xi * xi + xi +
               6. * eta * eta * eta * eta * eta + 5. * eta * eta * eta * eta + 4. * eta * eta * eta + 3. * eta * eta +
               2. * eta + 1.;
    };
    Approx o5_int = Approx(17.86666666667).margin(tol);

    SECTION("1 point quadrature")
    {
        const auto& quadrature = lstr::QuadratureGenerator< lstr::QuadratureTypes::GLeg, 1 >{}.get(element);

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
        const auto& quadrature = lstr::QuadratureGenerator< lstr::QuadratureTypes::GLeg, 3 >{}.get(element);

        REQUIRE(quadrature.size == 4);
        REQUIRE(quadrature.dim == 2);

        CHECK(lstr::invokeQuadrature(o0_fun, quadrature) == o0_int);
        CHECK(lstr::invokeQuadrature(o1_fun, quadrature) == o1_int);
        CHECK(lstr::invokeQuadrature(o2_fun, quadrature) == o2_int);
        CHECK(lstr::invokeQuadrature(o3_fun, quadrature) == o3_int);
    }

    SECTION("9 point quadrature")
    {
        const auto& quadrature = lstr::QuadratureGenerator< lstr::QuadratureTypes::GLeg, 5 >{}.get(element);

        REQUIRE(quadrature.size == 9);
        REQUIRE(quadrature.dim == 2);

        CHECK(lstr::invokeQuadrature(o0_fun, quadrature) == o0_int);
        CHECK(lstr::invokeQuadrature(o1_fun, quadrature) == o1_int);
        CHECK(lstr::invokeQuadrature(o2_fun, quadrature) == o2_int);
        CHECK(lstr::invokeQuadrature(o3_fun, quadrature) == o3_int);
        CHECK(lstr::invokeQuadrature(o4_fun, quadrature) == o4_int);
        CHECK(lstr::invokeQuadrature(o5_fun, quadrature) == o5_int);
    }
}

TEST_CASE("Gauss-Legendre quadratures for hexahedral element", "[quadrature]")
{
    const lstr::ElementData< lstr::ElementTypes::Hex, 1 > data{{lstr::Point{0., 0., 0.},
                                                                lstr::Point{0., 0., 0.},
                                                                lstr::Point{0., 0., 0.},
                                                                lstr::Point{0., 0., 0.},
                                                                lstr::Point{0., 0., 0.},
                                                                lstr::Point{0., 0., 0.},
                                                                lstr::Point{0., 0., 0.},
                                                                lstr::Point{0., 0., 0.}}};
    const auto element = lstr::Element< lstr::ElementTypes ::Hex, 1 >{{1, 2, 3, 4, 5, 6, 7, 8}, data, 0};

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
        const auto& quadrature = lstr::QuadratureGenerator< lstr::QuadratureTypes::GLeg, 1 >{}.get(element);

        REQUIRE(quadrature.size == 1);
        REQUIRE(quadrature.dim == 3);

        Approx qp_0 = Approx(0).margin(tol);
        Approx w_0  = Approx(8).margin(tol);
        CHECK(quadrature.getQPoints()[0][0] == qp_0);
        CHECK(quadrature.getQPoints()[0][1] == qp_0);
        CHECK(quadrature.getQPoints()[0][2] == qp_0);
        CHECK(quadrature.getWeights()[0] == w_0);
    }

    SECTION("8 point quadrature")
    {
        const auto& quadrature = lstr::QuadratureGenerator< lstr::QuadratureTypes::GLeg, 3 >{}.get(element);

        REQUIRE(quadrature.size == 8);
        REQUIRE(quadrature.dim == 3);

        CHECK(lstr::invokeQuadrature(o0_fun, quadrature) == o0_int);
        CHECK(lstr::invokeQuadrature(o1_fun, quadrature) == o1_int);
        CHECK(lstr::invokeQuadrature(o2_fun, quadrature) == o2_int);
        CHECK(lstr::invokeQuadrature(o3_fun, quadrature) == o3_int);
    }

    SECTION("512 point quadrature")
    {
        const auto& quadrature = lstr::QuadratureGenerator< lstr::QuadratureTypes::GLeg, 15 >{}.get(element);

        REQUIRE(quadrature.size == 512);
        REQUIRE(quadrature.dim == 3);

        CHECK(lstr::invokeQuadrature(trig_fun, quadrature) == trig_int);
    }
}
