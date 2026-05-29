#include "l3ster/mapping/BoundaryNormal.hpp"
#include "l3ster/mapping/ComputePhysicalDerivatives.hpp"
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
using namespace lstr::basis;

static auto getLineElement()
{
    return Element{{0, 1}, ElementData< ElementType::Line, 1 >{{Point{0., 0., 0.}, Point{1., 0., 0.}}}, 0};
}

static auto getQuadElement()
{
    using namespace lstr;
    return Element{{0, 1, 2, 3},
                   ElementData< ElementType::Quad, 1 >{
                       {Point{0., 0., 0.}, Point{1., 0., 0.}, Point{0., 1., 0.}, Point{2., 2., 0.}}},
                   0};
}

static auto getHexElement()
{
    return Element{{0, 1, 2, 3, 4, 5, 6, 7},
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

static auto approx(val_t a)
{
    constexpr auto eps = 1e-15;
    return Approx(a).margin(eps);
}

TEST_CASE("Reference to physical mapping", "[mesh]")
{
    constexpr auto BT = basis::BasisType::Lagrange;
    constexpr auto EO = 2;
    constexpr auto GO = 1;

    SECTION("1D")
    {
        constexpr auto ET       = ElementType::Line;
        using element_type      = Element< ET, EO >;
        constexpr auto el_nodes = element_type::node_array_t{};
        const auto     data     = ElementData< ET, EO >{{Point{1., 1., 1.}, Point{.5, .5, .5}}};
        const auto     element  = element_type{el_nodes, data, 0};
        constexpr auto point    = std::array{Point{0.}};
        const auto     basis    = basis::tabulateBasis< BT, ET, GO >(std::span{point});
        const auto     mapped   = mapToPhysicalSpace(basis.values, std::span{element.data.vertices});

        CHECK(mapped[0].x() == approx(.75));
        CHECK(mapped[0].y() == approx(.75));
        CHECK(mapped[0].z() == approx(.75));
    }

    SECTION("2D")
    {
        constexpr auto ET       = ElementType::Quad;
        using element_type      = Element< ET, EO >;
        constexpr auto el_nodes = element_type::node_array_t{};
        const auto     data =
            ElementData< ET, EO >{{Point{1., -1., 0.}, Point{2., -1., 0.}, Point{1., 1., 1.}, Point{2., 1., 1.}}};
        const auto     element = element_type{el_nodes, data, 0};
        constexpr auto point   = std::array{Point{.5, -.5}};
        const auto     basis   = basis::tabulateBasis< BT, ET, GO >(std::span{point});
        const auto     mapped  = mapToPhysicalSpace(basis.values, std::span{element.data.vertices});

        CHECK(mapped[0].x() == approx(1.75));
        CHECK(mapped[0].y() == approx(-.5));
        CHECK(mapped[0].z() == approx(.25));
    }

    SECTION("3D")
    {
        constexpr auto ET       = ElementType::Hex;
        using element_type      = Element< ET, EO >;
        constexpr auto el_nodes = element_type::node_array_t{};
        const auto     data     = ElementData< ET, EO >{{Point{.5, .5, .5},
                                                         Point{1., .5, .5},
                                                         Point{.5, 1., .5},
                                                         Point{1., 1., .5},
                                                         Point{.5, .5, 1.},
                                                         Point{1., .5, 1.},
                                                         Point{.5, 1., 1.},
                                                         Point{1., 1., 1.}}};
        const auto     element  = element_type{el_nodes, data, 0};
        constexpr auto point    = std::array{Point{0., 0., 0.}};
        const auto     basis    = basis::tabulateBasis< BT, ET, GO >(std::span{point});
        const auto     mapped   = mapToPhysicalSpace(basis.values, std::span{element.data.vertices});

        CHECK(mapped[0].x() == approx(.75));
        CHECK(mapped[0].y() == approx(.75));
        CHECK(mapped[0].z() == approx(.75));
    }
}

TEST_CASE("Jacobi matrix computation", "[mapping]")
{
    constexpr auto BT = basis::BasisType::Lagrange;
    constexpr auto EO = 1;

    SECTION("Line")
    {
        const auto     element = getLineElement();
        constexpr auto point   = std::array{Point{.42}};
        const auto     basis   = basis::tabulateBasis< BT, ElementType::Line, EO >(std::span{point});
        const auto     jm      = computeJacobiMats(basis.derivatives, std::span{element.data.vertices});
        CHECK(jm.elems[0][0] == approx(.5));
    }

    SECTION("Quad")
    {
        const auto     element = getQuadElement();
        constexpr auto point   = std::array{Point{.5, .5}};
        const auto     basis   = basis::tabulateBasis< BT, ElementType::Quad, EO >(std::span{point});
        const auto     jm      = computeJacobiMats(basis.derivatives, std::span{element.data.vertices});
        CHECK(jm.elems[0][0] == approx(7. / 8.));
        CHECK(jm.elems[1][0] == approx(3. / 8.));
        CHECK(jm.elems[2][0] == approx(3. / 8.));
        CHECK(jm.elems[3][0] == approx(7. / 8.));
    }

    SECTION("Hex")
    {
        const auto     element = getHexElement();
        constexpr auto point   = std::array{Point{.5, .5, .5}};
        const auto     basis   = basis::tabulateBasis< BT, ElementType::Hex, EO >(std::span{point});
        const auto     jm      = computeJacobiMats(basis.derivatives, std::span{element.data.vertices});
        CHECK(jm.elems[0][0] == approx(.5));
        CHECK(jm.elems[1][0] == approx(0.));
        CHECK(jm.elems[2][0] == approx(0.));
        CHECK(jm.elems[3][0] == approx(0.));
        CHECK(jm.elems[4][0] == approx(.5));
        CHECK(jm.elems[5][0] == approx(0.));
        CHECK(jm.elems[6][0] == approx(3. / 16.));
        CHECK(jm.elems[7][0] == approx(3. / 16.));
        CHECK(jm.elems[8][0] == approx(7. / 8.));
    }
}

TEST_CASE("Boundary normal computation", "[mapping]")
{
    constexpr auto BT = basis::BasisType::Lagrange;
    constexpr auto EO = 1;

    SECTION("Line")
    {
        constexpr auto ET           = ElementType::Line;
        const auto     element      = getLineElement();
        constexpr auto points       = std::array{Point{0.}, Point{1.}};
        const auto     basis        = basis::tabulateBasis< BT, ET, EO >(std::span{points});
        const auto     jm           = computeJacobiMats(basis.derivatives, std::span{element.data.vertices});
        const auto     left_normal  = computeBoundaryNormal< ET >(0, jm.get(0));
        const auto     right_normal = computeBoundaryNormal< ET >(1, jm.get(1));
        CHECK(left_normal[0] == approx(-1.));
        CHECK(right_normal[0] == approx(1.));
    }

    SECTION("Quad")
    {
        constexpr auto ET      = ElementType::Quad;
        const auto     element = getQuadElement();
        constexpr auto points  = std::array{Point{0., -1.}, Point{0., 1.}, Point{-1., 0.}, Point{1., 0.}};
        const auto     basis   = basis::tabulateBasis< BT, ET, EO >(std::span{points});
        const auto     jm      = computeJacobiMats(basis.derivatives, std::span{element.data.vertices});
        const auto     normals = util::elwise(util::makeIotaArray< dim_t, points.size() >(),
                                          [&](dim_t i) { return computeBoundaryNormal< ET >(i, jm.get(i)); });
        CHECK(normals[0][0] == approx(0.));
        CHECK(normals[0][1] == approx(-1.));
        CHECK(normals[1][0] == approx(-1. / std::sqrt(5.)));
        CHECK(normals[1][1] == approx(2. / std::sqrt(5.)));
        CHECK(normals[2][0] == approx(-1.));
        CHECK(normals[2][1] == approx(0.));
        CHECK(normals[3][0] == approx(2. / std::sqrt(5.)));
        CHECK(normals[3][1] == approx(-1. / std::sqrt(5.)));
    }

    SECTION("Hex")
    {
        constexpr auto ET      = ElementType::Hex;
        const auto     element = getHexElement();
        constexpr auto points  = std::array{Point{0., 0., -1.},
                                           Point{0., 0., 1.},
                                           Point{0., -1., 0.},
                                           Point{0., 1., 0.},
                                           Point{-1., 0., 0.},
                                           Point{1., 0., 0.}};
        const auto     basis   = basis::tabulateBasis< BT, ET, EO >(std::span{points});
        const auto     jm      = computeJacobiMats(basis.derivatives, std::span{element.data.vertices});
        const auto     normals = util::elwise(util::makeIotaArray< dim_t, points.size() >(),
                                          [&](dim_t i) { return computeBoundaryNormal< ET >(i, jm.get(i)); });
        CHECK(normals[0][0] == approx(0.));
        CHECK(normals[0][1] == approx(0.));
        CHECK(normals[0][2] == approx(-1.));
        CHECK(normals[1][0] == approx(-std::sqrt(1. / 6.)));
        CHECK(normals[1][1] == approx(-std::sqrt(1. / 6.)));
        CHECK(normals[1][2] == approx(std::sqrt(2. / 3.)));
        CHECK(normals[2][0] == approx(0.));
        CHECK(normals[2][1] == approx(-1.));
        CHECK(normals[2][2] == approx(0.));
        CHECK(normals[3][0] == approx(0.));
        CHECK(normals[3][1] == approx(1.));
        CHECK(normals[3][2] == approx(0.));
        CHECK(normals[4][0] == approx(-1.));
        CHECK(normals[4][1] == approx(0.));
        CHECK(normals[4][2] == approx(0.));
        CHECK(normals[5][0] == approx(1.));
        CHECK(normals[5][1] == approx(0.));
        CHECK(normals[5][2] == approx(0.));
    }
}

TEST_CASE("Basis function values", "[mapping]")
{
    using namespace basis;
    constexpr auto LB = BasisType::Lagrange;
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
    constexpr auto BT = BasisType::Lagrange;
    constexpr auto EO = 1;

    SECTION("Line")
    {
        constexpr auto ET      = ElementType::Line;
        const auto     element = getLineElement();
        constexpr auto points  = std::array{Point{0.}};
        const auto     basis   = basis::tabulateBasis< BT, ET, EO >(std::span{points});
        auto           jm      = computeJacobiMats(basis.derivatives, std::span{element.data.vertices});
        const auto     j       = computeJacobians(jm);
        invertJacobiMats(jm, std::span{j});
        const auto ders = computePhysicalDerivatives(basis.derivatives, jm);
        CHECK(ders.getMap(0)(0, 0) == approx(-1.));
        CHECK(ders.getMap(0)(0, 1) == approx(1.));
    }

    SECTION("Quad")
    {
        constexpr auto ET      = ElementType::Quad;
        const auto     element = getQuadElement();
        constexpr auto points  = std::array{Point{0., 0.}};
        const auto     basis   = basis::tabulateBasis< BT, ET, EO >(std::span{points});
        auto           jm      = computeJacobiMats(basis.derivatives, std::span{element.data.vertices});
        const auto     j       = computeJacobians(jm);
        invertJacobiMats(jm, std::span{j});
        const auto ders = computePhysicalDerivatives(basis.derivatives, jm);
        CHECK(ders.getMap(0)[0] == approx(-.25));
        CHECK(ders.getMap(1)[0] == approx(-.25));
        CHECK(ders.getMap(0)[1] == approx(.5));
        CHECK(ders.getMap(1)[1] == approx(-.5));
        CHECK(ders.getMap(0)[2] == approx(-.5));
        CHECK(ders.getMap(1)[2] == approx(.5));
        CHECK(ders.getMap(0)[3] == approx(.25));
        CHECK(ders.getMap(1)[3] == approx(.25));
    }

    SECTION("Hex")
    {
        constexpr auto ET      = ElementType::Hex;
        constexpr auto element = Element{{0, 1, 2, 3, 4, 5, 6, 7},
                                         ElementData< ElementType::Hex, 1 >{{Point{0., 0., 0.},
                                                                             Point{1., 0., 0.},
                                                                             Point{0., 1., 0.},
                                                                             Point{1., 1., 0.},
                                                                             Point{0., 0., 1.},
                                                                             Point{1., 0., 1.},
                                                                             Point{0., 1., 1.},
                                                                             Point{1., 1., 1.}}},
                                         0};
        constexpr auto points  = std::array{Point{0., 0., 0.}};
        const auto     basis   = basis::tabulateBasis< BT, ET, EO >(std::span{points});
        auto           jm      = computeJacobiMats(basis.derivatives, std::span{element.data.vertices});
        const auto     j       = computeJacobians(jm);
        invertJacobiMats(jm, std::span{j});
        const auto ders = computePhysicalDerivatives(basis.derivatives, jm);
        CHECK(ders.getMap(0)[0] == approx(-.25));
        CHECK(ders.getMap(1)[0] == approx(-.25));
        CHECK(ders.getMap(2)[0] == approx(-.25));
        CHECK(ders.getMap(0)[1] == approx(.25));
        CHECK(ders.getMap(1)[1] == approx(-.25));
        CHECK(ders.getMap(2)[1] == approx(-.25));
        CHECK(ders.getMap(0)[2] == approx(-.25));
        CHECK(ders.getMap(1)[2] == approx(.25));
        CHECK(ders.getMap(2)[2] == approx(-.25));
        CHECK(ders.getMap(0)[3] == approx(.25));
        CHECK(ders.getMap(1)[3] == approx(.25));
        CHECK(ders.getMap(2)[3] == approx(-.25));
        CHECK(ders.getMap(0)[4] == approx(-.25));
        CHECK(ders.getMap(1)[4] == approx(-.25));
        CHECK(ders.getMap(2)[4] == approx(.25));
        CHECK(ders.getMap(0)[5] == approx(.25));
        CHECK(ders.getMap(1)[5] == approx(-.25));
        CHECK(ders.getMap(2)[5] == approx(.25));
        CHECK(ders.getMap(0)[6] == approx(-.25));
        CHECK(ders.getMap(1)[6] == approx(.25));
        CHECK(ders.getMap(2)[6] == approx(.25));
        CHECK(ders.getMap(0)[7] == approx(.25));
        CHECK(ders.getMap(1)[7] == approx(.25));
        CHECK(ders.getMap(2)[7] == approx(.25));
    }
}

TEST_CASE("Reference basis at domain QPs", "[mapping]")
{
    constexpr auto   ET = ElementType::Hex;
    constexpr el_o_t EO = 4;
    constexpr auto   QT = quad::QuadratureType::GaussLegendre;
    constexpr el_o_t QO = 4;
    constexpr auto   BT = BasisType::Lagrange;

    const auto [bases, wgts]              = getQuadratureView< BT, ET, EO, QT, QO >();
    const auto [approx_basis, geom_basis] = bases;
    const auto np                         = decltype(approx_basis->getValuesMap())::RowsAtCompileTime;

    SECTION("Values")
    {
        for (int p = 0; p != np; ++p)
            CHECK(approx_basis->getValuesMap().row(p).sum() == approx(1.));
        for (int p = 0; p != np; ++p)
            CHECK(geom_basis->getValuesMap().row(p).sum() == approx(1.));
    }

    SECTION("Derivatives")
    {
        for (dim_t dim = 0; dim != 3; ++dim)
            for (int p = 0; p != np; ++p)
                CHECK(approx_basis->getDerivativesMap(dim).row(p).sum() == approx(0.));
        for (dim_t dim = 0; dim != 3; ++dim)
            for (int p = 0; p != np; ++p)
                CHECK(geom_basis->getDerivativesMap(dim).row(p).sum() == approx(0.));
    }
}

TEST_CASE("Reference basis at boundary QPs", "[mapping]")
{
    constexpr auto  QT = quad::QuadratureType::GaussLegendre;
    constexpr q_o_t QO = 5;
    constexpr auto  BT = BasisType::Lagrange;

    constexpr auto check_all_in_plane = []< el_o_t... orders >(
                                            const BoundaryView< orders... >& view, Space normal, val_t offs) {
        const auto element_checker = [&]< ElementType ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) {
            const auto [bases, wgts] = getSideQuadratureView< BT, ET, EO, QT, QO >(el_view.getSide());
            const auto points        = mapToPhysicalSpace(bases.geom_basis->values, std::span{el_view->data.vertices});
            for (auto qp : points)
                CHECK(qp[std::to_underlying(normal)] == approx(offs));
        };
        view.element_views.visit(element_checker, std::execution::seq);
    };

    SECTION("Generated")
    {
        const auto node_pos = std::array{0., .25, .5, .75, 1.};

        SECTION("1D")
        {
            constexpr auto   ET = ElementType::Line;
            constexpr el_o_t EO = 1;
            const auto       el =
                Element< ET, 1 >{std::array< n_id_t, 2 >{0, 1},
                                 std::array{Point{node_pos.front(), 0., 0.}, Point{node_pos.back(), 0., 0.}},
                                 0};

            const auto check_pos = [&](el_side_t side, val_t x_pos) {
                const auto [bases, wgts] = getSideQuadratureView< BT, ET, EO, QT, QO >(side);
                const auto points        = mapToPhysicalSpace(bases.geom_basis->values, std::span{el.data.vertices});
                CHECK(points[0][0] == approx(x_pos));
                CHECK(points[0][1] == approx(0.));
                CHECK(points[0][2] == approx(0.));
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
    constexpr auto  BT = BasisType::Lagrange;
    constexpr auto  QT = quad::QuadratureType::GaussLegendre;
    constexpr q_o_t QO = 10;

    const auto check_side_area = [&]< ElementType ET, el_o_t EO >(
                                     const Element< ET, EO >& element, el_side_t side, val_t expected_area) {
        constexpr auto params    = KernelParams{.dimension = Element< ET, EO >::native_dim, .n_equations = 1};
        constexpr auto integrand = wrapBoundaryResidualKernel< params >([](const auto&, auto& out) {
            out[0] = 1.; // Compute boundary area/length
        });
        constexpr auto GT        = ElementTraits< Element< ET, EO > >::geom_type;
        constexpr auto gt_wrp    = util::ConstexprValue< GT >{};

        const auto basis_at_qps = getSideQuadratureView< BT, ET, EO, QT, QO >(side);
        const auto verts        = std::span{element.data.vertices};
        const auto mapping      = TabulatedBoundaryMapping{basis_at_qps.bases, verts, gt_wrp, side};
        const auto node_vals    = Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, 0 >{};

        const auto area = post::evalElementBoundaryIntegral(integrand, mapping, node_vals, basis_at_qps.weights, 0.)[0];
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
