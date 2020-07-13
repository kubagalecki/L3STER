#include "catch2/catch.hpp"
#include "l3ster.h"

TEST_CASE("Quadratures are computed correctly", "[quadrature]")
{
    lstr::mesh::Element< lstr::mesh::ElementTypes::Quad, 1 >                element{{1, 2, 3, 4}};
    lstr::quad::QuadratureGenerator< lstr::quad::QuadratureTypes::GLeg, 1 > quad_gen;
    const auto& quad  = quad_gen.get(element);
    const auto& q_pts = quad.getQPoints();
    const auto& q_w   = quad.getWeights();

    REQUIRE(q_pts.size() == q_w.size());
}
