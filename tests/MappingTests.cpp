#include "l3ster.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

TEST_CASE("Jacobi matrix computation", "[mapping]")
{
    using namespace lstr;
    SECTION("Line")
    {
        Element< ElementTypes::Line, 1 > el{
            {0, 1}, ElementData< ElementTypes::Line, 1 >{{Point{0., 0., 0.}, Point{1., 0., 0.}}}, 0};
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(el);
        const auto val             = jacobi_mat_eval(Point{.42});
        CHECK(val[0] == Approx(.5).epsilon(1e-13));
    }

    SECTION("Quad")
    {
        Element< ElementTypes::Quad, 1 > el{
            {0, 1, 2, 3},
            ElementData< ElementTypes::Quad, 1 >{
                {Point{0., 0., 0.}, Point{1., 0., 0.}, Point{0., 1., 0.}, Point{2., 2., 0.}}},
            0};
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(el);
        const auto val             = jacobi_mat_eval(Point{.5, .5});
        CHECK(val(0, 0) == Approx(7. / 8.).epsilon(1e-13));
        CHECK(val(0, 1) == Approx(3. / 8.).epsilon(1e-13));
        CHECK(val(1, 0) == Approx(3. / 8.).epsilon(1e-13));
        CHECK(val(1, 1) == Approx(7. / 8.).epsilon(1e-13));
    }

    SECTION("Hex")
    {
        Element< ElementTypes::Hex, 1 > el{{0, 1, 2, 3, 4, 5, 6, 7},
                                           ElementData< ElementTypes::Hex, 1 >{{Point{0., 0., 0.},
                                                                                Point{1., 0., 0.},
                                                                                Point{0., 1., 0.},
                                                                                Point{1., 1., 0.},
                                                                                Point{0., 0., 1.},
                                                                                Point{1., 0., 1.},
                                                                                Point{0., 1., 1.},
                                                                                Point{2., 2., 2.}}},
                                           0};
        const auto                      jacobi_mat_eval = getNatJacobiMatGenerator(el);
        const auto                      val             = jacobi_mat_eval(Point{.5, .5, .5});
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
