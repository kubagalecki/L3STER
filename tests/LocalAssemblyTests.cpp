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
        for (ptrdiff_t basis = 0; basis < vals.rows(); ++basis)
            CHECK(vals(basis, Eigen::all).sum() == Approx{1.});
    }

    SECTION("Derivatives")
    {
        constexpr auto   ET   = ElementTypes::Hex;
        constexpr el_o_t EO   = 4;
        constexpr auto   QT   = QuadratureTypes::GLeg;
        constexpr el_o_t QO   = 4;
        const auto&      ders = getRefBasisDersAtQpoints< QT, QO, ET, EO, BasisTypes::Lagrange >();
        for (const auto& der : ders)
            for (ptrdiff_t basis = 0; basis < der.rows(); ++basis)
                CHECK(der(basis, Eigen::all).sum() == Approx{0.}.margin(1e-14));
    }
}