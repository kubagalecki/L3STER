#include "catch2/catch.hpp"

#include "l3ster/assembly/AssembleLocalMatrix.hpp"
#include "l3ster/mapping/ComputeBasisDerivative.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"

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

TEST_CASE("Basis at QPs physical derivative computation", "[local_asm, mapping]")
{
    constexpr auto  QT = QuadratureTypes::GLeg;
    constexpr q_o_t QO = 5;
    constexpr auto  BT = BasisTypes::Lagrange;

    const auto mesh = makeCubeMesh(std::vector{0., .25, .5, .75, 1.});
    const auto part = mesh.getPartitions()[0];
    part.cvisit(
        []< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
            const auto test_point = Point{getQuadrature< QT, QO, ET >().getPoints().front()};
            const auto J          = getNatJacobiMatGenerator(element)(test_point);
            const auto bas_ders_at_testp =
                computePhysBasisDers< ET, EO >(J, computeRefBasisDers< ET, EO, BT >(test_point));

            const auto ders = computePhysicalBasesAtQpoints< QT, QO, BT >(element);

            for (int i = 0; i < ElementTraits< Element< ET, EO > >::native_dim; ++i)
                CHECK((ders[i](0, Eigen::all) - bas_ders_at_testp(i, Eigen::all)).norm() == Approx{0.}.margin(1e-13));
        },
        {0}); // Only for the hex domain, this won't work for 2D elements in a 3D space
}

TEST_CASE("Assemble local matrix", "[local_asm]")
{
    // TODO
}
