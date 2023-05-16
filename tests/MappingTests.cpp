#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"
#include "l3ster/mapping/BoundaryNormal.hpp"
#include "l3ster/mapping/ComputePhysBasisDer.hpp"
#include "l3ster/mapping/JacobiMat.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"
#include "l3ster/mesh/ReadMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/mesh/primitives/LineMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/Integral.hpp"

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
                                                                                Point{1., 0., 1.5},
                                                                                Point{0., 1., 1.5},
                                                                                Point{1., 1., 2.}}},
                                           0};
}

TEST_CASE("Reference to physical mapping", "[mesh]")
{
    constexpr auto el_o = 2;
    SECTION("1D")
    {
        constexpr auto el_t                = ElementTypes::Line;
        using element_type                 = Element< el_t, el_o >;
        constexpr auto            el_nodes = typename element_type::node_array_t{};
        ElementData< el_t, el_o > data{{Point{1., 1., 1.}, Point{.5, .5, .5}}};
        const auto                element = element_type{el_nodes, data, 0};
        const auto                mapped  = mapToPhysicalSpace(element, Point{0.});
        CHECK(mapped.x() == Approx(.75).margin(1e-15));
        CHECK(mapped.y() == Approx(.75).margin(1e-15));
        CHECK(mapped.z() == Approx(.75).margin(1e-15));
    }

    SECTION("2D")
    {
        constexpr auto el_t                = ElementTypes::Quad;
        using element_type                 = Element< el_t, el_o >;
        constexpr auto            el_nodes = typename element_type::node_array_t{};
        ElementData< el_t, el_o > data{{Point{1., -1., 0.}, Point{2., -1., 0.}, Point{1., 1., 1.}, Point{2., 1., 1.}}};
        const auto                element = element_type{el_nodes, data, 0};
        const auto                mapped  = mapToPhysicalSpace(element, Point{.5, -.5});
        CHECK(mapped.x() == Approx(1.75).margin(1e-15));
        CHECK(mapped.y() == Approx(-.5).margin(1e-15));
        CHECK(mapped.z() == Approx(.25).margin(1e-15));
    }

    SECTION("3D")
    {
        constexpr auto el_t                = ElementTypes::Hex;
        using element_type                 = Element< el_t, el_o >;
        constexpr auto            el_nodes = typename element_type::node_array_t{};
        ElementData< el_t, el_o > data{{Point{.5, .5, .5},
                                        Point{1., .5, .5},
                                        Point{.5, 1., .5},
                                        Point{1., 1., .5},
                                        Point{.5, .5, 1.},
                                        Point{1., .5, 1.},
                                        Point{.5, 1., 1.},
                                        Point{1., 1., 1.}}};
        const auto                element = element_type{el_nodes, data, 0};
        const auto                mapped  = mapToPhysicalSpace(element, Point{0., 0., 0.});
        CHECK(mapped.x() == Approx(.75).margin(1e-15));
        CHECK(mapped.y() == Approx(.75).margin(1e-15));
        CHECK(mapped.z() == Approx(.75).margin(1e-15));
    }
}

TEST_CASE("Jacobi matrix computation", "[mapping]")
{
    SECTION("Line")
    {
        const auto element         = getLineElement();
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element);
        const auto val             = jacobi_mat_eval(Point{.42});
        CHECK(val[0] == Approx(.5).margin(1e-13));
    }

    SECTION("Quad")
    {
        const auto element         = getQuadElement();
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element);
        const auto val             = jacobi_mat_eval(Point{.5, .5});
        CHECK(val(0, 0) == Approx(7. / 8.).margin(1e-13));
        CHECK(val(0, 1) == Approx(3. / 8.).margin(1e-13));
        CHECK(val(1, 0) == Approx(3. / 8.).margin(1e-13));
        CHECK(val(1, 1) == Approx(7. / 8.).margin(1e-13));
    }

    SECTION("Hex")
    {
        const auto element         = getHexElement();
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element);
        const auto val             = jacobi_mat_eval(Point{.5, .5, .5});
        CHECK(val(0, 0) == Approx(.5).margin(1e-13));
        CHECK(val(0, 1) == Approx(0.).margin(1e-13));
        CHECK(val(0, 2) == Approx(3. / 16.).margin(1e-13));
        CHECK(val(1, 0) == Approx(0.).margin(1e-13));
        CHECK(val(1, 1) == Approx(.5).margin(1e-13));
        CHECK(val(1, 2) == Approx(3. / 16.).margin(1e-13));
        CHECK(val(2, 0) == Approx(0.).margin(1e-13));
        CHECK(val(2, 1) == Approx(0.).margin(1e-13));
        CHECK(val(2, 2) == Approx(7. / 8.).margin(1e-13));
    }
}

TEST_CASE("Boundary normal computation", "[mapping]")
{
    SECTION("Line")
    {
        const auto element         = getLineElement();
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element);
        const auto left_view       = BoundaryElementView{element, 0};
        const auto right_view      = BoundaryElementView{element, 1};
        const auto left_normal     = computeBoundaryNormal(left_view, jacobi_mat_eval(Point{0.}));
        const auto right_normal    = computeBoundaryNormal(right_view, jacobi_mat_eval(Point{1.}));
        CHECK(left_normal[0] == Approx(-1.).margin(1e-13));
        CHECK(right_normal[0] == Approx(1.).margin(1e-13));
    }
    SECTION("Quad")
    {
        const auto element         = getQuadElement();
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element);
        const auto bot_view        = BoundaryElementView{element, 0};
        const auto top_view        = BoundaryElementView{element, 1};
        const auto left_view       = BoundaryElementView{element, 2};
        const auto right_view      = BoundaryElementView{element, 3};
        const auto bot_normal      = computeBoundaryNormal(bot_view, jacobi_mat_eval(Point{0., -1.}));
        const auto top_normal      = computeBoundaryNormal(top_view, jacobi_mat_eval(Point{0., 1.}));
        const auto left_normal     = computeBoundaryNormal(left_view, jacobi_mat_eval(Point{-1., 0.}));
        const auto right_normal    = computeBoundaryNormal(right_view, jacobi_mat_eval(Point{1., 0.}));
        CHECK(bot_normal[0] == Approx(0.).margin(1e-13));
        CHECK(bot_normal[1] == Approx(-1.).margin(1e-13));
        CHECK(top_normal[0] == Approx(-1. / std::sqrt(5.)).margin(1e-13));
        CHECK(top_normal[1] == Approx(2. / std::sqrt(5.)).margin(1e-13));
        CHECK(left_normal[0] == Approx(-1.).margin(1e-13));
        CHECK(left_normal[1] == Approx(0.).margin(1e-13));
        CHECK(right_normal[0] == Approx(2. / std::sqrt(5.)).margin(1e-13));
        CHECK(right_normal[1] == Approx(-1. / std::sqrt(5.)).margin(1e-13));
    }
    SECTION("Hex")
    {
        const auto element         = getHexElement();
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element);
        const auto front_view      = BoundaryElementView{element, 0};
        const auto back_view       = BoundaryElementView{element, 1};
        const auto bot_view        = BoundaryElementView{element, 2};
        const auto top_view        = BoundaryElementView{element, 3};
        const auto left_view       = BoundaryElementView{element, 4};
        const auto right_view      = BoundaryElementView{element, 5};
        const auto front_normal    = computeBoundaryNormal(front_view, jacobi_mat_eval(Point{0., 0., -1.}));
        const auto back_normal     = computeBoundaryNormal(back_view, jacobi_mat_eval(Point{0., 0., 1.}));
        const auto bot_normal      = computeBoundaryNormal(bot_view, jacobi_mat_eval(Point{0., -1., 0.}));
        const auto top_normal      = computeBoundaryNormal(top_view, jacobi_mat_eval(Point{0., 1., 0.}));
        const auto left_normal     = computeBoundaryNormal(left_view, jacobi_mat_eval(Point{-1., 0., 0.}));
        const auto right_normal    = computeBoundaryNormal(right_view, jacobi_mat_eval(Point{1., 0., 0.}));
        CHECK(front_normal[0] == Approx(0.).margin(1e-13));
        CHECK(front_normal[1] == Approx(0.).margin(1e-13));
        CHECK(front_normal[2] == Approx(-1.).margin(1e-13));
        CHECK(back_normal[0] == Approx(-std::sqrt(1. / 6.)).margin(1e-13));
        CHECK(back_normal[1] == Approx(-std::sqrt(1. / 6.)).margin(1e-13));
        CHECK(back_normal[2] == Approx(std::sqrt(2. / 3.)).margin(1e-13));
        CHECK(bot_normal[0] == Approx(0.).margin(1e-13));
        CHECK(bot_normal[1] == Approx(-1.).margin(1e-13));
        CHECK(bot_normal[2] == Approx(0.).margin(1e-13));
        CHECK(top_normal[0] == Approx(0.).margin(1e-13));
        CHECK(top_normal[1] == Approx(1.).margin(1e-13));
        CHECK(top_normal[2] == Approx(0.).margin(1e-13));
        CHECK(left_normal[0] == Approx(-1.).margin(1e-13));
        CHECK(left_normal[1] == Approx(0.).margin(1e-13));
        CHECK(left_normal[2] == Approx(0.).margin(1e-13));
        CHECK(right_normal[0] == Approx(1.).margin(1e-13));
        CHECK(right_normal[1] == Approx(0.).margin(1e-13));
        CHECK(right_normal[2] == Approx(0.).margin(1e-13));
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

        CHECK(basis0(Point{-1.}) == Approx{1.}.margin(1e-13));
        CHECK(basis0(Point{1.}) == Approx{0.}.margin(1e-13));
        CHECK(basis1(Point{-1.}) == Approx{0.}.margin(1e-13));
        CHECK(basis1(Point{1.}) == Approx{1.}.margin(1e-13));
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

        CHECK(basis0(p0) == Approx{.75 * .75}.margin(1e-13));
        CHECK(basis0(p1) == Approx{.25 * .25}.margin(1e-13));
        CHECK(basis0(p2) == Approx{0.}.margin(1e-13));

        CHECK(basis1(p0) == Approx{.25 * .75}.margin(1e-13));
        CHECK(basis1(p1) == Approx{.25 * .75}.margin(1e-13));
        CHECK(basis1(p2) == Approx{0.}.margin(1e-13));

        CHECK(basis2(p0) == Approx{.25 * .75}.margin(1e-13));
        CHECK(basis2(p1) == Approx{.25 * .75}.margin(1e-13));
        CHECK(basis2(p2) == Approx{0.}.margin(1e-13));

        CHECK(basis3(p0) == Approx{.25 * .25}.margin(1e-13));
        CHECK(basis3(p1) == Approx{.75 * .75}.margin(1e-13));
        CHECK(basis3(p2) == Approx{1.}.margin(1e-13));
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

        CHECK(basis0(p0) == Approx{.75 * .75 * .75}.margin(1e-13));
        CHECK(basis0(p1) == Approx{.25 * .25 * .25}.margin(1e-13));
        CHECK(basis0(p2) == Approx{0.}.margin(1e-13));
        CHECK(basis0(p3) == Approx{0.}.margin(1e-13));

        CHECK(basis1(p0) == Approx{.25 * .75 * .75}.margin(1e-13));
        CHECK(basis1(p1) == Approx{.25 * .25 * .75}.margin(1e-13));
        CHECK(basis1(p2) == Approx{0.}.margin(1e-13));
        CHECK(basis1(p3) == Approx{0.}.margin(1e-13));

        CHECK(basis2(p0) == Approx{.25 * .75 * .75}.margin(1e-13));
        CHECK(basis2(p1) == Approx{.25 * .25 * .75}.margin(1e-13));
        CHECK(basis2(p2) == Approx{0.}.margin(1e-13));
        CHECK(basis2(p3) == Approx{0.}.margin(1e-13));

        CHECK(basis3(p0) == Approx{.25 * .25 * .75}.margin(1e-13));
        CHECK(basis3(p1) == Approx{.25 * .75 * .75}.margin(1e-13));
        CHECK(basis3(p2) == Approx{1.}.margin(1e-13));
        CHECK(basis3(p3) == Approx{0.}.margin(1e-13));

        CHECK(basis4(p0) == Approx{.25 * .75 * .75}.margin(1e-13));
        CHECK(basis4(p1) == Approx{.25 * .25 * .75}.margin(1e-13));
        CHECK(basis4(p2) == Approx{0.}.margin(1e-13));
        CHECK(basis4(p3) == Approx{0.}.margin(1e-13));

        CHECK(basis5(p0) == Approx{.25 * .25 * .75}.margin(1e-13));
        CHECK(basis5(p1) == Approx{.25 * .75 * .75}.margin(1e-13));
        CHECK(basis5(p2) == Approx{0.}.margin(1e-13));
        CHECK(basis5(p3) == Approx{0.}.margin(1e-13));

        CHECK(basis6(p0) == Approx{.25 * .25 * .75}.margin(1e-13));
        CHECK(basis6(p1) == Approx{.25 * .75 * .75}.margin(1e-13));
        CHECK(basis6(p2) == Approx{0.}.margin(1e-13));
        CHECK(basis6(p3) == Approx{.5}.margin(1e-13));

        CHECK(basis7(p0) == Approx{.25 * .25 * .25}.margin(1e-13));
        CHECK(basis7(p1) == Approx{.75 * .75 * .75}.margin(1e-13));
        CHECK(basis7(p2) == Approx{0.}.margin(1e-13));
        CHECK(basis7(p3) == Approx{.5}.margin(1e-13));
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
        CHECK(bf_der(0, 0) == Approx(-1.).margin(1e-13));
        CHECK(bf_der(0, 1) == Approx(1.).margin(1e-13));
    }

    SECTION("Quad")
    {
        const auto element    = getQuadElement();
        const auto test_point = Point{0., 0.};
        const auto jac        = getNatJacobiMatGenerator(element)(test_point);
        const auto ref_ders = computeRefBasisDers< decltype(element)::type, decltype(element)::order, LB >(test_point);
        const auto bf_der   = computePhysBasisDers(jac, ref_ders);
        CHECK(bf_der(0, 0) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(1, 0) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(0, 1) == Approx(.5).margin(1e-13));
        CHECK(bf_der(1, 1) == Approx(-.5).margin(1e-13));
        CHECK(bf_der(0, 2) == Approx(-.5).margin(1e-13));
        CHECK(bf_der(1, 2) == Approx(.5).margin(1e-13));
        CHECK(bf_der(0, 3) == Approx(.25).margin(1e-13));
        CHECK(bf_der(1, 3) == Approx(.25).margin(1e-13));
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
        CHECK(bf_der(0, 0) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(1, 0) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(2, 0) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(0, 1) == Approx(.25).margin(1e-13));
        CHECK(bf_der(1, 1) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(2, 1) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(0, 2) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(1, 2) == Approx(.25).margin(1e-13));
        CHECK(bf_der(2, 2) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(0, 3) == Approx(.25).margin(1e-13));
        CHECK(bf_der(1, 3) == Approx(.25).margin(1e-13));
        CHECK(bf_der(2, 3) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(0, 4) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(1, 4) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(2, 4) == Approx(.25).margin(1e-13));
        CHECK(bf_der(0, 5) == Approx(.25).margin(1e-13));
        CHECK(bf_der(1, 5) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(2, 5) == Approx(.25).margin(1e-13));
        CHECK(bf_der(0, 6) == Approx(-.25).margin(1e-13));
        CHECK(bf_der(1, 6) == Approx(.25).margin(1e-13));
        CHECK(bf_der(2, 6) == Approx(.25).margin(1e-13));
        CHECK(bf_der(0, 7) == Approx(.25).margin(1e-13));
        CHECK(bf_der(1, 7) == Approx(.25).margin(1e-13));
        CHECK(bf_der(2, 7) == Approx(.25).margin(1e-13));
    }
}

TEST_CASE("Reference basis at domain QPs", "[mapping]")
{
    constexpr auto   ET            = ElementTypes::Hex;
    constexpr el_o_t EO            = 4;
    constexpr auto   QT            = QuadratureTypes::GLeg;
    constexpr el_o_t QO            = 4;
    constexpr auto   BT            = BasisTypes::Lagrange;
    const auto       ref_bas_at_qp = getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();

    SECTION("Values")
    {
        for (const auto& vals_at_qp : ref_bas_at_qp.basis.values)
            CHECK(vals_at_qp.sum() == Approx{1.});
    }

    SECTION("Derivatives")
    {
        for (const auto& ders_at_qp : ref_bas_at_qp.basis.derivatives)
            for (Eigen::Index dim = 0; dim < ders_at_qp.rows(); ++dim)
                CHECK(ders_at_qp(dim, Eigen::all).sum() == Approx{0.}.margin(1e-13));
    }
}

TEST_CASE("Reference basis at boundary QPs", "[mapping]")
{
    constexpr auto  QT = QuadratureTypes::GLeg;
    constexpr q_o_t QO = 5;
    constexpr auto  BT = BasisTypes::Lagrange;

    constexpr auto check_all_in_plane = [](const BoundaryView& view, Space normal, val_t offs) {
        const auto space_ind = std::invoke(
            [](Space s) {
                int retval{};
                switch (s)
                {
                case Space::X:
                    retval = 0;
                    break;
                case Space::Y:
                    retval = 1;
                    break;
                case Space::Z:
                    retval = 2;
                    break;
                }
                return retval;
            },
            normal);
        const auto element_checker = [&]< ElementTypes ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) {
            const auto& ref_q =
                getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(el_view.getSide()).quadrature;
            for (auto qp : ref_q.points)
                CHECK(mapToPhysicalSpace(*el_view, qp)[space_ind] == Approx{offs}.margin(1.e-15));
        };
        view.visit(element_checker);
    };

    SECTION("Generated")
    {
        const auto node_pos = std::array{0., .25, .5, .75, 1.};

        SECTION("1D")
        {
            constexpr auto   ET  = ElementTypes::Line;
            constexpr el_o_t EO  = 1;
            constexpr q_o_t  QLO = 1;
            const auto       el  = Element< ET, 1 >{
                std::array< n_id_t, 2 >{0, 1},
                std::array< Point< 3 >, 2 >{Point{node_pos.front(), 0., 0.}, Point{node_pos.back(), 0., 0.}},
                0};

            const auto check_pos = [&](el_side_t side, val_t x_pos) {
                const auto& ref_q  = getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QLO >(side);
                const auto& ref_p  = ref_q.quadrature.points.front();
                const auto  phys_p = mapToPhysicalSpace(el, Point{ref_p});
                CHECK(phys_p[0] == Approx{x_pos}.margin(1e-15));
                CHECK(phys_p[1] == Approx{0.}.margin(1e-15));
                CHECK(phys_p[2] == Approx{0.}.margin(1e-15));
            };
            check_pos(0, 0.);
            check_pos(1, 1.);
        }
        SECTION("2D")
        {
            const auto mesh = makeSquareMesh(node_pos);

            const auto b_bottom = mesh.getBoundaryView(1);
            const auto b_top    = mesh.getBoundaryView(2);
            const auto b_left   = mesh.getBoundaryView(3);
            const auto b_right  = mesh.getBoundaryView(4);

            check_all_in_plane(b_bottom, Space::Y, node_pos.front());
            check_all_in_plane(b_top, Space::Y, node_pos.back());
            check_all_in_plane(b_left, Space::X, node_pos.front());
            check_all_in_plane(b_right, Space::X, node_pos.back());
        }
        SECTION("3D")
        {
            const auto mesh = makeCubeMesh(node_pos);

            const auto b_front  = mesh.getBoundaryView(1);
            const auto b_back   = mesh.getBoundaryView(2);
            const auto b_bottom = mesh.getBoundaryView(3);
            const auto b_top    = mesh.getBoundaryView(4);
            const auto b_left   = mesh.getBoundaryView(5);
            const auto b_right  = mesh.getBoundaryView(6);

            check_all_in_plane(b_front, Space::Z, node_pos.front());
            check_all_in_plane(b_back, Space::Z, node_pos.back());
            check_all_in_plane(b_bottom, Space::Y, node_pos.front());
            check_all_in_plane(b_top, Space::Y, node_pos.back());
            check_all_in_plane(b_left, Space::X, node_pos.front());
            check_all_in_plane(b_right, Space::X, node_pos.back());
        }
    }
    SECTION("Read from gmsh")
    {
        SECTION("2D")
        {
            const auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), gmsh_tag);

            const auto b_bottom = mesh.getBoundaryView(5);
            const auto b_top    = mesh.getBoundaryView(3);
            const auto b_left   = mesh.getBoundaryView(2);
            const auto b_right  = mesh.getBoundaryView(4);

            check_all_in_plane(b_bottom, Space::Y, -.5);
            check_all_in_plane(b_top, Space::Y, .5);
            check_all_in_plane(b_left, Space::X, -.5);
            check_all_in_plane(b_right, Space::X, .5);
        }
        SECTION("3D")
        {
            const auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube.msh), gmsh_tag);

            const auto b_front  = mesh.getBoundaryView(2);
            const auto b_back   = mesh.getBoundaryView(3);
            const auto b_bottom = mesh.getBoundaryView(4);
            const auto b_top    = mesh.getBoundaryView(5);
            const auto b_left   = mesh.getBoundaryView(7);
            const auto b_right  = mesh.getBoundaryView(6);

            check_all_in_plane(b_front, Space::Z, -1.);
            check_all_in_plane(b_back, Space::Z, 1.);
            check_all_in_plane(b_bottom, Space::Y, -1.);
            check_all_in_plane(b_top, Space::Y, 1.);
            check_all_in_plane(b_left, Space::X, -1.);
            check_all_in_plane(b_right, Space::X, 1.);
        }
    }
}

TEST_CASE("Boundary integration", "[mapping]")
{
    constexpr auto  BT        = BasisTypes::Lagrange;
    constexpr auto  QT        = QuadratureTypes::GLeg;
    constexpr q_o_t QO        = 10;
    constexpr auto  integrand = [](const auto&, const auto&, const auto&, const auto&) noexcept {
        return Eigen::Vector< val_t, 1 >(1.); // Compute boundary area/length
    };
    const auto check_side_area =
        [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element, el_side_t side, val_t expected_area) {
            const auto el_view    = BoundaryElementView{element, side};
            const auto basis_vals = getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(side);
            const auto node_vals  = eigen::RowMajorMatrix< val_t, Element< ET, EO >::n_nodes, 0 >{};
            const auto area = detail::evalElementBoundaryIntegral(integrand, el_view, node_vals, basis_vals, 0.)[0];
            CHECK(area == Approx(expected_area).margin(1e-15));
        };

    SECTION("1D")
    {
        const auto element = getLineElement();
        check_side_area(element, 0, 0.);
        check_side_area(element, 1, 0.);
    }

    SECTION("2D")
    {
        const auto element = getQuadElement();
        check_side_area(element, 0, 1.);
        check_side_area(element, 1, std::sqrt(5.));
        check_side_area(element, 2, 1.);
        check_side_area(element, 3, std::sqrt(5.));
    }

    SECTION("3D")
    {
        const auto element = getHexElement();
        check_side_area(element, 0, 1.);
        check_side_area(element, 1, std::sqrt(1.5));
        check_side_area(element, 2, 1.25);
        check_side_area(element, 3, 1.75);
        check_side_area(element, 4, 1.25);
        check_side_area(element, 5, 1.75);
    }
}
