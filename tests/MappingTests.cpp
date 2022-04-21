#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"
#include "l3ster/mapping/ComputePhysBasisDer.hpp"
#include "l3ster/mapping/ComputePhysBasisDersAtQpoints.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

using namespace lstr;

static auto getLineElement()
{
    return Element< ElementTypes::Line, 1 >{
        {0, 1}, ElementData< ElementTypes::Line, 1 >{{Point{0., 0., 0.}, Point{1., 0., 0.}}}, 0};
}

static auto getQuadElement()
{
    using namespace lstr;
    return Element< ElementTypes::Quad, 1 >{
        {0, 1, 2, 3},
        ElementData< ElementTypes::Quad, 1 >{
            {Point{0., 0., 0.}, Point{1., 0., 0.}, Point{0., 1., 0.}, Point{2., 2., 0.}}},
        0};
}

static auto getHexElement()
{
    return Element< ElementTypes::Hex, 1 >{{0, 1, 2, 3, 4, 5, 6, 7},
                                           ElementData< ElementTypes::Hex, 1 >{{Point{0., 0., 0.},
                                                                                Point{1., 0., 0.},
                                                                                Point{0., 1., 0.},
                                                                                Point{1., 1., 0.},
                                                                                Point{0., 0., 1.},
                                                                                Point{1., 0., 1.},
                                                                                Point{0., 1., 1.},
                                                                                Point{2., 2., 2.}}},
                                           0};
}

TEST_CASE("Jacobi matrix computation", "[mapping]")
{
    SECTION("Line")
    {
        const auto element         = getLineElement();
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element);
        const auto val             = jacobi_mat_eval(Point{.42});
        CHECK(val[0] == Approx(.5).epsilon(1e-13));
    }

    SECTION("Quad")
    {
        const auto element         = getQuadElement();
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element);
        const auto val             = jacobi_mat_eval(Point{.5, .5});
        CHECK(val(0, 0) == Approx(7. / 8.).epsilon(1e-13));
        CHECK(val(0, 1) == Approx(3. / 8.).epsilon(1e-13));
        CHECK(val(1, 0) == Approx(3. / 8.).epsilon(1e-13));
        CHECK(val(1, 1) == Approx(7. / 8.).epsilon(1e-13));
    }

    SECTION("Hex")
    {
        const auto element         = getHexElement();
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element);
        const auto val             = jacobi_mat_eval(Point{.5, .5, .5});
        CHECK(val(0, 0) == Approx(25. / 32.).epsilon(1e-13));
        CHECK(val(0, 1) == Approx(9. / 32.).epsilon(1e-13));
        CHECK(val(0, 2) == Approx(9. / 32.).epsilon(1e-13));
        CHECK(val(1, 0) == Approx(9. / 32.).epsilon(1e-13));
        CHECK(val(1, 1) == Approx(25. / 32.).epsilon(1e-13));
        CHECK(val(1, 2) == Approx(9. / 32.).epsilon(1e-13));
        CHECK(val(2, 0) == Approx(9. / 32.).epsilon(1e-13));
        CHECK(val(2, 1) == Approx(9. / 32.).epsilon(1e-13));
        CHECK(val(2, 2) == Approx(25. / 32.).epsilon(1e-13));
    }
}

TEST_CASE("Basis function values", "[mapping]")
{
    constexpr auto LB = BasisTypes::Lagrange;
    SECTION("Line")
    {
        constexpr auto   ET = ElementTypes::Line;
        constexpr el_o_t EO = 1;

        const ReferenceBasisFunction< ET, EO, 0, LB > basis0{};
        const ReferenceBasisFunction< ET, EO, 1, LB > basis1{};

        CHECK(basis0(Point{-1.}) == Approx{1.}.epsilon(1e-13));
        CHECK(basis0(Point{1.}) == Approx{0.}.epsilon(1e-13));
        CHECK(basis1(Point{-1.}) == Approx{0.}.epsilon(1e-13));
        CHECK(basis1(Point{1.}) == Approx{1.}.epsilon(1e-13));
    }

    SECTION("Quad")
    {
        constexpr auto   ET = ElementTypes::Quad;
        constexpr el_o_t EO = 1;

        const ReferenceBasisFunction< ET, EO, 0, LB > basis0{};
        const ReferenceBasisFunction< ET, EO, 1, LB > basis1{};
        const ReferenceBasisFunction< ET, EO, 2, LB > basis2{};
        const ReferenceBasisFunction< ET, EO, 3, LB > basis3{};

        const Point p0{-.5, -.5};
        const Point p1{.5, .5};
        const Point p2{1., 1.};

        CHECK(basis0(p0) == Approx{.75 * .75}.epsilon(1e-13));
        CHECK(basis0(p1) == Approx{.25 * .25}.epsilon(1e-13));
        CHECK(basis0(p2) == Approx{0.}.epsilon(1e-13));

        CHECK(basis1(p0) == Approx{.25 * .75}.epsilon(1e-13));
        CHECK(basis1(p1) == Approx{.25 * .75}.epsilon(1e-13));
        CHECK(basis1(p2) == Approx{0.}.epsilon(1e-13));

        CHECK(basis2(p0) == Approx{.25 * .75}.epsilon(1e-13));
        CHECK(basis2(p1) == Approx{.25 * .75}.epsilon(1e-13));
        CHECK(basis2(p2) == Approx{0.}.epsilon(1e-13));

        CHECK(basis3(p0) == Approx{.25 * .25}.epsilon(1e-13));
        CHECK(basis3(p1) == Approx{.75 * .75}.epsilon(1e-13));
        CHECK(basis3(p2) == Approx{1.}.epsilon(1e-13));
    }

    SECTION("Hex")
    {
        constexpr auto   ET = ElementTypes::Hex;
        constexpr el_o_t EO = 1;

        const ReferenceBasisFunction< ET, EO, 0, LB > basis0{};
        const ReferenceBasisFunction< ET, EO, 1, LB > basis1{};
        const ReferenceBasisFunction< ET, EO, 2, LB > basis2{};
        const ReferenceBasisFunction< ET, EO, 3, LB > basis3{};
        const ReferenceBasisFunction< ET, EO, 4, LB > basis4{};
        const ReferenceBasisFunction< ET, EO, 5, LB > basis5{};
        const ReferenceBasisFunction< ET, EO, 6, LB > basis6{};
        const ReferenceBasisFunction< ET, EO, 7, LB > basis7{};

        const Point p0{-.5, -.5, -.5};
        const Point p1{.5, .5, .5};
        const Point p2{1., 1., -1.};
        const Point p3{0., 1., 1.};

        CHECK(basis0(p0) == Approx{.75 * .75 * .75}.epsilon(1e-13));
        CHECK(basis0(p1) == Approx{.25 * .25 * .25}.epsilon(1e-13));
        CHECK(basis0(p2) == Approx{0.}.epsilon(1e-13));
        CHECK(basis0(p3) == Approx{0.}.epsilon(1e-13));

        CHECK(basis1(p0) == Approx{.25 * .75 * .75}.epsilon(1e-13));
        CHECK(basis1(p1) == Approx{.25 * .25 * .75}.epsilon(1e-13));
        CHECK(basis1(p2) == Approx{0.}.epsilon(1e-13));
        CHECK(basis1(p3) == Approx{0.}.epsilon(1e-13));

        CHECK(basis2(p0) == Approx{.25 * .75 * .75}.epsilon(1e-13));
        CHECK(basis2(p1) == Approx{.25 * .25 * .75}.epsilon(1e-13));
        CHECK(basis2(p2) == Approx{0.}.epsilon(1e-13));
        CHECK(basis2(p3) == Approx{0.}.epsilon(1e-13));

        CHECK(basis3(p0) == Approx{.25 * .25 * .75}.epsilon(1e-13));
        CHECK(basis3(p1) == Approx{.25 * .75 * .75}.epsilon(1e-13));
        CHECK(basis3(p2) == Approx{1.}.epsilon(1e-13));
        CHECK(basis3(p3) == Approx{0.}.epsilon(1e-13));

        CHECK(basis4(p0) == Approx{.25 * .75 * .75}.epsilon(1e-13));
        CHECK(basis4(p1) == Approx{.25 * .25 * .75}.epsilon(1e-13));
        CHECK(basis4(p2) == Approx{0.}.epsilon(1e-13));
        CHECK(basis4(p3) == Approx{0.}.epsilon(1e-13));

        CHECK(basis5(p0) == Approx{.25 * .25 * .75}.epsilon(1e-13));
        CHECK(basis5(p1) == Approx{.25 * .75 * .75}.epsilon(1e-13));
        CHECK(basis5(p2) == Approx{0.}.epsilon(1e-13));
        CHECK(basis5(p3) == Approx{0.}.epsilon(1e-13));

        CHECK(basis6(p0) == Approx{.25 * .25 * .75}.epsilon(1e-13));
        CHECK(basis6(p1) == Approx{.25 * .75 * .75}.epsilon(1e-13));
        CHECK(basis6(p2) == Approx{0.}.epsilon(1e-13));
        CHECK(basis6(p3) == Approx{.5}.epsilon(1e-13));

        CHECK(basis7(p0) == Approx{.25 * .25 * .25}.epsilon(1e-13));
        CHECK(basis7(p1) == Approx{.75 * .75 * .75}.epsilon(1e-13));
        CHECK(basis7(p2) == Approx{0.}.epsilon(1e-13));
        CHECK(basis7(p3) == Approx{.5}.epsilon(1e-13));
    }
}

TEST_CASE("Basis function derivatives", "[mapping]")
{
    constexpr auto LB = BasisTypes::Lagrange;
    SECTION("Line")
    {
        const auto element    = getLineElement();
        const auto test_point = Point{0.};
        const auto jac        = getNatJacobiMatGenerator(element)(test_point);
        const auto ref_ders = computeRefBasisDers< decltype(element)::type, decltype(element)::order, LB >(test_point);
        const auto bf_der   = computePhysBasisDers(jac, ref_ders);
        CHECK(bf_der(0, 0) == Approx(-1.).epsilon(1e-13));
        CHECK(bf_der(0, 1) == Approx(1.).epsilon(1e-13));
    }

    SECTION("Quad")
    {
        const auto element    = getQuadElement();
        const auto test_point = Point{0., 0.};
        const auto jac        = getNatJacobiMatGenerator(element)(test_point);
        const auto ref_ders = computeRefBasisDers< decltype(element)::type, decltype(element)::order, LB >(test_point);
        const auto bf_der   = computePhysBasisDers(jac, ref_ders);
        CHECK(bf_der(0, 0) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(1, 0) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(0, 1) == Approx(.5).epsilon(1e-13));
        CHECK(bf_der(1, 1) == Approx(-.5).epsilon(1e-13));
        CHECK(bf_der(0, 2) == Approx(-.5).epsilon(1e-13));
        CHECK(bf_der(1, 2) == Approx(.5).epsilon(1e-13));
        CHECK(bf_der(0, 3) == Approx(.25).epsilon(1e-13));
        CHECK(bf_der(1, 3) == Approx(.25).epsilon(1e-13));
    }

    SECTION("Hex")
    {
        const auto element    = Element< ElementTypes::Hex, 1 >{{0, 1, 2, 3, 4, 5, 6, 7},
                                                                ElementData< ElementTypes::Hex, 1 >{{Point{0., 0., 0.},
                                                                                                     Point{1., 0., 0.},
                                                                                                     Point{0., 1., 0.},
                                                                                                     Point{1., 1., 0.},
                                                                                                     Point{0., 0., 1.},
                                                                                                     Point{1., 0., 1.},
                                                                                                     Point{0., 1., 1.},
                                                                                                     Point{1., 1., 1.}}},
                                                                0};
        const auto test_point = Point{0., 0., 0.};
        const auto jac        = getNatJacobiMatGenerator(element)(test_point);
        const auto ref_ders = computeRefBasisDers< decltype(element)::type, decltype(element)::order, LB >(test_point);
        const auto bf_der   = computePhysBasisDers(jac, ref_ders);
        CHECK(bf_der(0, 0) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(1, 0) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(2, 0) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(0, 1) == Approx(.25).epsilon(1e-13));
        CHECK(bf_der(1, 1) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(2, 1) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(0, 2) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(1, 2) == Approx(.25).epsilon(1e-13));
        CHECK(bf_der(2, 2) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(0, 3) == Approx(.25).epsilon(1e-13));
        CHECK(bf_der(1, 3) == Approx(.25).epsilon(1e-13));
        CHECK(bf_der(2, 3) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(0, 4) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(1, 4) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(2, 4) == Approx(.25).epsilon(1e-13));
        CHECK(bf_der(0, 5) == Approx(.25).epsilon(1e-13));
        CHECK(bf_der(1, 5) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(2, 5) == Approx(.25).epsilon(1e-13));
        CHECK(bf_der(0, 6) == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der(1, 6) == Approx(.25).epsilon(1e-13));
        CHECK(bf_der(2, 6) == Approx(.25).epsilon(1e-13));
        CHECK(bf_der(0, 7) == Approx(.25).epsilon(1e-13));
        CHECK(bf_der(1, 7) == Approx(.25).epsilon(1e-13));
        CHECK(bf_der(2, 7) == Approx(.25).epsilon(1e-13));
    }
}

TEST_CASE("Reference basis at QPs", "[mapping]")
{
    constexpr auto   ET            = ElementTypes::Hex;
    constexpr el_o_t EO            = 4;
    constexpr auto   QT            = QuadratureTypes::GLeg;
    constexpr el_o_t QO            = 4;
    constexpr auto   BT            = BasisTypes::Lagrange;
    const auto       ref_bas_at_qp = getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();

    SECTION("Values")
    {
        for (ptrdiff_t basis = 0; basis < ref_bas_at_qp.basis_vals.rows(); ++basis)
            CHECK(ref_bas_at_qp.basis_vals(basis, Eigen::all).sum() == Approx{1.});
    }

    SECTION("Derivatives")
    {
        for (const auto& der : ref_bas_at_qp.basis_ders)
            for (ptrdiff_t basis = 0; basis < der.rows(); ++basis)
                CHECK(der(basis, Eigen::all).sum() == Approx{0.}.margin(1e-13));
    }
}

TEST_CASE("Physical basis derivatives at QPs", "[mapping]")
{
    constexpr auto  QT = QuadratureTypes::GLeg;
    constexpr q_o_t QO = 5;
    constexpr auto  BT = BasisTypes::Lagrange;

    constexpr auto do_test = []< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        const auto test_point        = Point{getQuadrature< QT, QO, ET >().getPoints().front()};
        const auto J                 = getNatJacobiMatGenerator(element)(test_point);
        const auto ref_basis_ders    = computeRefBasisDers< ET, EO, BT >(test_point);
        const auto bas_ders_at_testp = computePhysBasisDers(J, ref_basis_ders);

        const auto& ref_basis_at_qps = getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
        const auto  jacobians_at_qps = computeJacobiansAtQpoints(element, ref_basis_at_qps.quadrature);
        const auto  phys_ders_at_qps = computePhysBasisDersAtQpoints(ref_basis_at_qps.basis_ders, jacobians_at_qps);

        for (int i = 0; i < Element< ET, EO >::native_dim; ++i)
            CHECK((phys_ders_at_qps[i](0, Eigen::all) - bas_ders_at_testp(i, Eigen::all)).norm() ==
                  Approx{0.}.margin(1e-13));
    };

    const auto mesh = makeCubeMesh(std::vector{0., .25, .5, .75, 1.});
    const auto part = mesh.getPartitions()[0];
    part.cvisit(do_test, {0}); // Only for the hex domain, this won't work for 2D elements in a 3D space
}
