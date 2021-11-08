#include "catch2/catch.hpp"

#include "l3ster/assembly/ComputeRefBasesAtQpoints.hpp"

using namespace lstr;

TEST_CASE("Reference bases at QPs test", "[local_asm]")
{
    SECTION("Values")
    {
        constexpr auto   ET   = ElementTypes::Hex;
        constexpr el_o_t EO   = 4;
        constexpr auto   QT   = QuadratureTypes::GLeg;
        constexpr el_o_t QO   = 4;
        const auto&      vals = getRefBasesAtQpoints< QT, QO, ET, EO, BasisTypes::Lagrange >();
        for (ptrdiff_t i = 0; i < vals.rows(); ++i)
            CHECK(vals(i, Eigen::all).sum() == Approx{1.}.epsilon(1e-10));
    }
}