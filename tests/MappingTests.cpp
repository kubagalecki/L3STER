#include "l3ster/mapping/ComputeBasisDerivative.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

static auto getLineElement()
{
    return lstr::Element< lstr::ElementTypes::Line, 1 >{
        {0, 1},
        lstr::ElementData< lstr::ElementTypes::Line, 1 >{{lstr::Point{0., 0., 0.}, lstr::Point{1., 0., 0.}}},
        0};
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
    using namespace lstr;
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
    using namespace lstr;
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

TEST_CASE("Basis function derivatives", "[mapping]")
{
    using namespace lstr;
    SECTION("Line")
    {
        const auto element = getLineElement();
        const auto bf_der0 = computePhysBasisDers< 0 >(element, Point{0.});
        const auto bf_der1 = computePhysBasisDers< 1 >(element, Point{0.});
        CHECK(bf_der0[0] == Approx(-1.).epsilon(1e-13));
        CHECK(bf_der1[0] == Approx(1.).epsilon(1e-13));
    }

    SECTION("Quad")
    {
        const auto element = getQuadElement();
        const auto point   = Point{0., 0.};
        const auto bf_der0 = computePhysBasisDers< 0 >(element, point);
        const auto bf_der1 = computePhysBasisDers< 1 >(element, point);
        const auto bf_der2 = computePhysBasisDers< 2 >(element, point);
        const auto bf_der3 = computePhysBasisDers< 3 >(element, point);
        CHECK(bf_der0[0] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der0[1] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der1[0] == Approx(.5).epsilon(1e-13));
        CHECK(bf_der1[1] == Approx(-.5).epsilon(1e-13));
        CHECK(bf_der2[0] == Approx(-.5).epsilon(1e-13));
        CHECK(bf_der2[1] == Approx(.5).epsilon(1e-13));
        CHECK(bf_der3[0] == Approx(.25).epsilon(1e-13));
        CHECK(bf_der3[1] == Approx(.25).epsilon(1e-13));
    }

    SECTION("Hex")
    {
        const auto element = Element< ElementTypes::Hex, 1 >{{0, 1, 2, 3, 4, 5, 6, 7},
                                                             ElementData< ElementTypes::Hex, 1 >{{Point{0., 0., 0.},
                                                                                                  Point{1., 0., 0.},
                                                                                                  Point{0., 1., 0.},
                                                                                                  Point{1., 1., 0.},
                                                                                                  Point{0., 0., 1.},
                                                                                                  Point{1., 0., 1.},
                                                                                                  Point{0., 1., 1.},
                                                                                                  Point{1., 1., 1.}}},
                                                             0};
        const auto point   = Point{0., 0., 0.};
        const auto bf_der0 = computePhysBasisDers< 0 >(element, point);
        const auto bf_der1 = computePhysBasisDers< 1 >(element, point);
        const auto bf_der2 = computePhysBasisDers< 2 >(element, point);
        const auto bf_der3 = computePhysBasisDers< 3 >(element, point);
        const auto bf_der4 = computePhysBasisDers< 4 >(element, point);
        const auto bf_der5 = computePhysBasisDers< 5 >(element, point);
        const auto bf_der6 = computePhysBasisDers< 6 >(element, point);
        const auto bf_der7 = computePhysBasisDers< 7 >(element, point);
        CHECK(bf_der0[0] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der0[1] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der0[2] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der1[0] == Approx(.25).epsilon(1e-13));
        CHECK(bf_der1[1] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der1[2] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der2[0] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der2[1] == Approx(.25).epsilon(1e-13));
        CHECK(bf_der2[2] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der3[0] == Approx(.25).epsilon(1e-13));
        CHECK(bf_der3[1] == Approx(.25).epsilon(1e-13));
        CHECK(bf_der3[2] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der4[0] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der4[1] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der4[2] == Approx(.25).epsilon(1e-13));
        CHECK(bf_der5[0] == Approx(.25).epsilon(1e-13));
        CHECK(bf_der5[1] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der5[2] == Approx(.25).epsilon(1e-13));
        CHECK(bf_der6[0] == Approx(-.25).epsilon(1e-13));
        CHECK(bf_der6[1] == Approx(.25).epsilon(1e-13));
        CHECK(bf_der6[2] == Approx(.25).epsilon(1e-13));
        CHECK(bf_der7[0] == Approx(.25).epsilon(1e-13));
        CHECK(bf_der7[1] == Approx(.25).epsilon(1e-13));
        CHECK(bf_der7[2] == Approx(.25).epsilon(1e-13));
    }

    SECTION("Single consistent with aggregate")
    {
        constexpr auto check_basis_ders =
            []< ElementTypes T, el_o_t O >(const Element< T, O >&                                       element,
                                           const Point< ElementTraits< Element< T, O > >::native_dim >& point) {
                const auto jacobi_gen = getNatJacobiMatGenerator(element);
                const auto J          = jacobi_gen(point);
                const auto all_ders   = computePhysBasisDers< T, O >(J, computeRefBasisDers< T, O >(point));
                forConstexpr(
                    [&]< el_locind_t I >(std::integral_constant< el_locind_t, I >) {
                        const auto single_der = computePhysBasisDers< I >(element, point);
                        for (size_t i = 0; i < ElementTraits< Element< T, O > >::native_dim; ++i)
                            CHECK(single_der[i] == Approx(all_ders(i, I)).epsilon(1e-13));
                    },
                    std::make_integer_sequence< el_locind_t, Element< T, O >::n_nodes >{});
            };

        SECTION("Line")
        {
            const auto element = getLineElement();
            check_basis_ders(element, Point{0.});
        }

        SECTION("Quad")
        {
            const auto element = getQuadElement();
            check_basis_ders(element, Point{0., 0.});
        }

        SECTION("Hex")
        {
            const auto element = getHexElement();
            check_basis_ders(element, Point{0., 0., 0.});
        }
    }
}
