#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"
#include "l3ster/mapping/BoundaryNormal.hpp"
#include "l3ster/mapping/ComputePhysBasisDer.hpp"
#include "l3ster/mapping/JacobiMat.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"
#include "l3ster/mesh/ReadMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/Integral.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

using namespace lstr;
using namespace lstr::map;
using namespace lstr::mesh;

static auto getLineElement()
{
    return Element< ElementType::Line, 1 >{
        {0, 1}, ElementData< ElementType::Line, 1 >{{Point{0., 0., 0.}, Point{1., 0., 0.}}}, 0};
}

static auto getQuadElement()
{
    using namespace lstr;
    return Element< ElementType::Quad, 1 >{
        {0, 1, 2, 3},
        ElementData< ElementType::Quad, 1 >{
            {Point{0., 0., 0.}, Point{1., 0., 0.}, Point{0., 1., 0.}, Point{2., 2., 0.}}},
        0};
}

static auto getHexElement()
{
    return Element< ElementType::Hex, 1 >{{0, 1, 2, 3, 4, 5, 6, 7},
                                          ElementData< ElementType::Hex, 1 >{{Point{0., 0., 0.},
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
        constexpr auto el_t                = ElementType::Line;
        using element_type                 = Element< el_t, el_o >;
        constexpr auto            el_nodes = typename element_type::node_array_t{};
        ElementData< el_t, el_o > data{{Point{1., 1., 1.}, Point{.5, .5, .5}}};
        const auto                element = element_type{el_nodes, data, 0};
        const auto                mapped  = mapToPhysicalSpace(element.data, Point{0.});
        CHECK(mapped.x() == Approx(.75).margin(1e-15));
        CHECK(mapped.y() == Approx(.75).margin(1e-15));
        CHECK(mapped.z() == Approx(.75).margin(1e-15));
    }

    SECTION("2D")
    {
        constexpr auto el_t                = ElementType::Quad;
        using element_type                 = Element< el_t, el_o >;
        constexpr auto            el_nodes = typename element_type::node_array_t{};
        ElementData< el_t, el_o > data{{Point{1., -1., 0.}, Point{2., -1., 0.}, Point{1., 1., 1.}, Point{2., 1., 1.}}};
        const auto                element = element_type{el_nodes, data, 0};
        const auto                mapped  = mapToPhysicalSpace(element.data, Point{.5, -.5});
        CHECK(mapped.x() == Approx(1.75).margin(1e-15));
        CHECK(mapped.y() == Approx(-.5).margin(1e-15));
        CHECK(mapped.z() == Approx(.25).margin(1e-15));
    }

    SECTION("3D")
    {
        constexpr auto el_t                = ElementType::Hex;
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
        const auto                mapped  = mapToPhysicalSpace(element.data, Point{0., 0., 0.});
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
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element.data);
        const auto val             = jacobi_mat_eval(Point{.42});
        CHECK(val[0] == Approx(.5).margin(1e-13));
    }

    SECTION("Quad")
    {
        const auto element         = getQuadElement();
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element.data);
        const auto val             = jacobi_mat_eval(Point{.5, .5});
        CHECK(val(0, 0) == Approx(7. / 8.).margin(1e-13));
        CHECK(val(0, 1) == Approx(3. / 8.).margin(1e-13));
        CHECK(val(1, 0) == Approx(3. / 8.).margin(1e-13));
        CHECK(val(1, 1) == Approx(7. / 8.).margin(1e-13));
    }

    SECTION("Hex")
    {
        const auto element         = getHexElement();
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element.data);
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
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element.data);
        const auto compute_normal  = [&]< ElementType ET, el_o_t EO >(const BoundaryElementView< ET, EO >& bev,
                                                                     const Point< 1 >&                    point) {
            return computeBoundaryNormal< ET, EO >(bev.getSide(), jacobi_mat_eval(point));
        };
        const auto left_view    = BoundaryElementView{&element, 0};
        const auto right_view   = BoundaryElementView{&element, 1};
        const auto left_normal  = compute_normal(left_view, Point{0.});
        const auto right_normal = compute_normal(right_view, Point{1.});
        CHECK(left_normal[0] == Approx(-1.).margin(1e-13));
        CHECK(right_normal[0] == Approx(1.).margin(1e-13));
    }
    SECTION("Quad")
    {
        const auto element         = getQuadElement();
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element.data);
        const auto compute_normal  = [&]< ElementType ET, el_o_t EO >(const BoundaryElementView< ET, EO >& bev,
                                                                     const Point< 2 >&                    point) {
            return computeBoundaryNormal< ET, EO >(bev.getSide(), jacobi_mat_eval(point));
        };
        const auto bot_view     = BoundaryElementView{&element, 0};
        const auto top_view     = BoundaryElementView{&element, 1};
        const auto left_view    = BoundaryElementView{&element, 2};
        const auto right_view   = BoundaryElementView{&element, 3};
        const auto bot_normal   = compute_normal(bot_view, Point{0., -1.});
        const auto top_normal   = compute_normal(top_view, Point{0., 1.});
        const auto left_normal  = compute_normal(left_view, Point{-1., 0.});
        const auto right_normal = compute_normal(right_view, Point{1., 0.});
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
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element.data);
        const auto compute_normal  = [&]< ElementType ET, el_o_t EO >(const BoundaryElementView< ET, EO >& bev,
                                                                     const Point< 3 >&                    point) {
            return computeBoundaryNormal< ET, EO >(bev.getSide(), jacobi_mat_eval(point));
        };
        const auto front_view   = BoundaryElementView{&element, 0};
        const auto back_view    = BoundaryElementView{&element, 1};
        const auto bot_view     = BoundaryElementView{&element, 2};
        const auto top_view     = BoundaryElementView{&element, 3};
        const auto left_view    = BoundaryElementView{&element, 4};
        const auto right_view   = BoundaryElementView{&element, 5};
        const auto front_normal = compute_normal(front_view, Point{0., 0., -1.});
        const auto back_normal  = compute_normal(back_view, Point{0., 0., 1.});
        const auto bot_normal   = compute_normal(bot_view, Point{0., -1., 0.});
        const auto top_normal   = compute_normal(top_view, Point{0., 1., 0.});
        const auto left_normal  = compute_normal(left_view, Point{-1., 0., 0.});
        const auto right_normal = compute_normal(right_view, Point{1., 0., 0.});
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
    using namespace basis;
    constexpr auto LB     = BasisType::Lagrange;
    constexpr auto approx = [](val_t x) {
        return Approx{x}.margin(1e-16);
    };
    SECTION("Line")
    {
        constexpr auto   ET = ElementType::Line;
        constexpr el_o_t EO = 1;

        constexpr auto p1 = Point{-1.};
        constexpr auto p2 = Point{1.};

        const auto basis_vals1 = computeReferenceBases< ET, EO, LB >(p1);
        const auto basis_vals2 = computeReferenceBases< ET, EO, LB >(p2);

        CHECK(basis_vals1[0] == approx(1.));
        CHECK(basis_vals2[0] == approx(0.));
        CHECK(basis_vals1[1] == approx(0.));
        CHECK(basis_vals2[1] == approx(1.));
    }

    SECTION("Quad")
    {
        constexpr auto   ET = ElementType::Quad;
        constexpr el_o_t EO = 1;

        constexpr auto p1 = Point{-.5, -.5};
        constexpr auto p2 = Point{.5, .5};
        constexpr auto p3 = Point{1., 1.};

        const auto basis_vals1 = computeReferenceBases< ET, EO, LB >(p1);
        const auto basis_vals2 = computeReferenceBases< ET, EO, LB >(p2);
        const auto basis_vals3 = computeReferenceBases< ET, EO, LB >(p3);

        CHECK(basis_vals1[0] == approx(.75 * .75));
        CHECK(basis_vals1[1] == approx(.25 * .75));
        CHECK(basis_vals1[2] == approx(.75 * .25));
        CHECK(basis_vals1[3] == approx(.25 * .25));

        CHECK(basis_vals2[0] == approx(.25 * .25));
        CHECK(basis_vals2[1] == approx(.25 * .75));
        CHECK(basis_vals2[2] == approx(.75 * .25));
        CHECK(basis_vals2[3] == approx(.75 * .75));

        CHECK(basis_vals3[0] == approx(0.));
        CHECK(basis_vals3[1] == approx(0.));
        CHECK(basis_vals3[2] == approx(0.));
        CHECK(basis_vals3[3] == approx(1.));
    }

    SECTION("Hex")
    {
        constexpr auto   ET = ElementType::Hex;
        constexpr el_o_t EO = 1;

        constexpr auto p0 = Point{-.5, -.5, -.5};
        constexpr auto p1 = Point{.5, .5, .5};
        constexpr auto p2 = Point{1., 1., -1.};
        constexpr auto p3 = Point{0., 1., 1.};

        const auto basis_vals1 = computeReferenceBases< ET, EO, LB >(p0);
        const auto basis_vals2 = computeReferenceBases< ET, EO, LB >(p1);
        const auto basis_vals3 = computeReferenceBases< ET, EO, LB >(p2);
        const auto basis_vals4 = computeReferenceBases< ET, EO, LB >(p3);

        CHECK(basis_vals1[0] == approx(.75 * .75 * .75));
        CHECK(basis_vals1[1] == approx(.25 * .75 * .75));
        CHECK(basis_vals1[2] == approx(.75 * .25 * .75));
        CHECK(basis_vals1[3] == approx(.25 * .25 * .75));
        CHECK(basis_vals1[4] == approx(.75 * .75 * .25));
        CHECK(basis_vals1[5] == approx(.25 * .75 * .25));
        CHECK(basis_vals1[6] == approx(.75 * .25 * .25));
        CHECK(basis_vals1[7] == approx(.25 * .25 * .25));

        CHECK(basis_vals2[0] == approx(.25 * .25 * .25));
        CHECK(basis_vals2[1] == approx(.25 * .25 * .75));
        CHECK(basis_vals2[2] == approx(.25 * .25 * .75));
        CHECK(basis_vals2[3] == approx(.25 * .75 * .75));
        CHECK(basis_vals2[4] == approx(.25 * .25 * .75));
        CHECK(basis_vals2[5] == approx(.25 * .75 * .75));
        CHECK(basis_vals2[6] == approx(.25 * .75 * .75));
        CHECK(basis_vals2[7] == approx(.75 * .75 * .75));

        CHECK(basis_vals3[0] == approx(0.));
        CHECK(basis_vals3[1] == approx(0.));
        CHECK(basis_vals3[2] == approx(0.));
        CHECK(basis_vals3[3] == approx(1.));
        CHECK(basis_vals3[4] == approx(0.));
        CHECK(basis_vals3[5] == approx(0.));
        CHECK(basis_vals3[6] == approx(0.));
        CHECK(basis_vals3[7] == approx(0.));

        CHECK(basis_vals4[0] == approx(0.));
        CHECK(basis_vals4[1] == approx(0.));
        CHECK(basis_vals4[2] == approx(0.));
        CHECK(basis_vals4[3] == approx(0.));
        CHECK(basis_vals4[4] == approx(0.));
        CHECK(basis_vals4[5] == approx(0.));
        CHECK(basis_vals4[6] == approx(.5));
        CHECK(basis_vals4[7] == approx(.5));
    }
}

TEST_CASE("Basis function derivatives", "[mapping]")
{
    using namespace basis;
    constexpr auto LB = BasisType::Lagrange;
    SECTION("Line")
    {
        const auto element    = getLineElement();
        const auto test_point = Point{0.};
        const auto jac        = getNatJacobiMatGenerator(element.data)(test_point);
        const auto ref_ders =
            computeReferenceBasisDerivatives< decltype(element)::type, decltype(element)::order, LB >(test_point);
        const auto bf_der = computePhysBasisDers(jac, ref_ders);
        CHECK(bf_der(0, 0) == Approx(-1.).margin(1e-13));
        CHECK(bf_der(0, 1) == Approx(1.).margin(1e-13));
    }

    SECTION("Quad")
    {
        const auto element    = getQuadElement();
        const auto test_point = Point{0., 0.};
        const auto jac        = getNatJacobiMatGenerator(element.data)(test_point);
        const auto ref_ders =
            computeReferenceBasisDerivatives< decltype(element)::type, decltype(element)::order, LB >(test_point);
        const auto bf_der = computePhysBasisDers(jac, ref_ders);
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
        const auto element    = Element< ElementType::Hex, 1 >{{0, 1, 2, 3, 4, 5, 6, 7},
                                                               ElementData< ElementType::Hex, 1 >{{Point{0., 0., 0.},
                                                                                                   Point{1., 0., 0.},
                                                                                                   Point{0., 1., 0.},
                                                                                                   Point{1., 1., 0.},
                                                                                                   Point{0., 0., 1.},
                                                                                                   Point{1., 0., 1.},
                                                                                                   Point{0., 1., 1.},
                                                                                                   Point{1., 1., 1.}}},
                                                               0};
        const auto test_point = Point{0., 0., 0.};
        const auto jac        = getNatJacobiMatGenerator(element.data)(test_point);
        const auto ref_ders =
            computeReferenceBasisDerivatives< decltype(element)::type, decltype(element)::order, LB >(test_point);
        const auto bf_der = computePhysBasisDers(jac, ref_ders);
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
    using namespace basis;
    constexpr auto   ET            = ElementType::Hex;
    constexpr el_o_t EO            = 4;
    constexpr auto   QT            = quad::QuadratureType::GaussLegendre;
    constexpr el_o_t QO            = 4;
    constexpr auto   BT            = BasisType::Lagrange;
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
    constexpr auto  QT = quad::QuadratureType::GaussLegendre;
    constexpr q_o_t QO = 5;
    constexpr auto  BT = basis::BasisType::Lagrange;

    constexpr auto check_all_in_plane = []< el_o_t... orders >(
                                            const BoundaryView< orders... >& view, Space normal, val_t offs) {
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
        const auto element_checker = [&]< ElementType ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) {
            const auto& ref_q =
                basis::getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(el_view.getSide()).quadrature;
            for (auto qp : ref_q.points)
                CHECK(mapToPhysicalSpace(el_view->data, qp)[space_ind] == Approx{offs}.margin(1e-15));
        };
        view.element_views.visit(element_checker, std::execution::seq);
    };

    SECTION("Generated")
    {
        const auto node_pos = std::array{0., .25, .5, .75, 1.};

        SECTION("1D")
        {
            constexpr auto   ET  = ElementType::Line;
            constexpr el_o_t EO  = 1;
            constexpr q_o_t  QLO = 1;
            const auto       el  = Element< ET, 1 >{
                std::array< n_id_t, 2 >{0, 1},
                std::array< Point< 3 >, 2 >{Point{node_pos.front(), 0., 0.}, Point{node_pos.back(), 0., 0.}},
                0};

            const auto check_pos = [&](el_side_t side, val_t x_pos) {
                const auto& ref_q  = basis::getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QLO >(side);
                const auto& ref_p  = ref_q.quadrature.points.front();
                const auto  phys_p = mapToPhysicalSpace(el.data, Point{ref_p});
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

            const auto& b_bottom = mesh.getBoundary(1);
            const auto& b_top    = mesh.getBoundary(2);
            const auto& b_left   = mesh.getBoundary(3);
            const auto& b_right  = mesh.getBoundary(4);

            check_all_in_plane(b_bottom, Space::Y, node_pos.front());
            check_all_in_plane(b_top, Space::Y, node_pos.back());
            check_all_in_plane(b_left, Space::X, node_pos.front());
            check_all_in_plane(b_right, Space::X, node_pos.back());
        }
        SECTION("3D")
        {
            const auto mesh = makeCubeMesh(node_pos);

            const auto& b_front  = mesh.getBoundary(1);
            const auto& b_back   = mesh.getBoundary(2);
            const auto& b_bottom = mesh.getBoundary(3);
            const auto& b_top    = mesh.getBoundary(4);
            const auto& b_left   = mesh.getBoundary(5);
            const auto& b_right  = mesh.getBoundary(6);

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
            const auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), {2, 3, 4, 5}, gmsh_tag);

            const auto& b_bottom = mesh.getBoundary(5);
            const auto& b_top    = mesh.getBoundary(3);
            const auto& b_left   = mesh.getBoundary(2);
            const auto& b_right  = mesh.getBoundary(4);

            check_all_in_plane(b_bottom, Space::Y, -.5);
            check_all_in_plane(b_top, Space::Y, .5);
            check_all_in_plane(b_left, Space::X, -.5);
            check_all_in_plane(b_right, Space::X, .5);
        }
        SECTION("3D")
        {
            const auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube.msh), {2, 3, 4, 5, 6, 7}, gmsh_tag);

            const auto& b_front  = mesh.getBoundary(2);
            const auto& b_back   = mesh.getBoundary(3);
            const auto& b_bottom = mesh.getBoundary(4);
            const auto& b_top    = mesh.getBoundary(5);
            const auto& b_left   = mesh.getBoundary(7);
            const auto& b_right  = mesh.getBoundary(6);

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
    constexpr auto  BT = basis::BasisType::Lagrange;
    constexpr auto  QT = quad::QuadratureType::GaussLegendre;
    constexpr q_o_t QO = 10;

    const auto check_side_area =
        [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >& element, el_side_t side, val_t expected_area) {
            constexpr auto params    = KernelParams{.dimension = Element< ET, EO >::native_dim, .n_equations = 1};
            constexpr auto integrand = wrapBoundaryResidualKernel< params >([](const auto&, auto& out) {
                out[0] = 1.; // Compute boundary area/length
            });

            const auto el_view    = BoundaryElementView{&element, side};
            const auto basis_vals = basis::getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(side);
            const auto node_vals  = util::eigen::RowMajorMatrix< val_t, Element< ET, EO >::n_nodes, 0 >{};
            const auto area       = post::evalElementBoundaryIntegral(integrand, el_view, node_vals, basis_vals, 0.)[0];
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
