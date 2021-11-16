#include "catch2/catch.hpp"

#include "l3ster/assembly/AssembleLocalMatrix.hpp"
#include "l3ster/mapping/ComputeBasisDerivative.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
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

TEST_CASE("Physical basis derivative at QPs", "[local_asm, mapping]")
{
    constexpr auto  QT = QuadratureTypes::GLeg;
    constexpr q_o_t QO = 5;
    constexpr auto  BT = BasisTypes::Lagrange;

    constexpr auto do_test = []< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        const auto test_point        = Point{getQuadrature< QT, QO, ET >().getPoints().front()};
        const auto J                 = getNatJacobiMatGenerator(element)(test_point);
        const auto bas_ders_at_testp = computePhysBasisDers< ET, EO >(J, computeRefBasisDers< ET, EO, BT >(test_point));

        const auto ders = computePhysicalBasesAtQpoints< QT, QO, BT >(element);

        for (int i = 0; i < ElementTraits< Element< ET, EO > >::native_dim; ++i)
            CHECK((ders[i](0, Eigen::all) - bas_ders_at_testp(i, Eigen::all)).norm() == Approx{0.}.margin(1e-13));
    };

    const auto mesh = makeCubeMesh(std::vector{0., .25, .5, .75, 1.});
    const auto part = mesh.getPartitions()[0];
    part.cvisit(do_test, {0}); // Only for the hex domain, this won't work for 2D elements in a 3D space
}
#include <iomanip>
#include <iostream>
TEST_CASE("Assemble local matrix", "[local_asm]")
{
    SECTION("Diffusion 3D")
    {
        constexpr el_o_t EO   = 2;
        auto             mesh = makeCubeMesh(std::vector{.5, 1.});
        auto&            part = mesh.getPartitions()[0];
        part.initDualGraph();
        const auto ho_part = convertMeshToOrder< EO >(part);

        constexpr auto do_test = []< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
            if constexpr (Element< T, O >::native_dim != 3)
                return;
            else
            {
                constexpr size_t nf  = 4;
                constexpr size_t ne  = 7;
                constexpr size_t dim = 3;

                constexpr auto diffusion_kernel_3d = []() noexcept {
                    using A_t   = Eigen::Matrix< val_t, ne, nf >;
                    using F_t   = Eigen::Matrix< val_t, ne, 1 >;
                    using ret_t = std::pair< std::array< A_t, dim + 1 >, F_t >;
                    ret_t ret_val;
                    auto& [A0, A1, A2, A3] = ret_val.first;
                    auto& F                = ret_val.second;
                    for (auto& mat : ret_val.first)
                        mat.setZero();
                    F.setZero();

                    constexpr double lambda = 1.;

                    A0(1, 1) = -1.;
                    A0(2, 2) = -1.;
                    A0(3, 3) = -1.;

                    A1(0, 1) = lambda;
                    A1(1, 0) = 1.;
                    A1(5, 3) = -1.;
                    A1(6, 2) = 1.;

                    A2(0, 2) = lambda;
                    A2(2, 0) = 1.;
                    A2(4, 3) = 1.;
                    A2(6, 1) = -1.;

                    A3(0, 3) = lambda;
                    A3(3, 0) = 1.;
                    A3(4, 2) = -1.;
                    A3(5, 1) = 1.;

                    return ret_val;
                };

                constexpr auto  QT     = QuadratureTypes::GLeg;
                constexpr q_o_t QO     = 2 * (EO - 1);
                constexpr auto  BT     = BasisTypes::Lagrange;
                auto            system = assembleLocalMatrix< QT, QO, BT >(diffusion_kernel_3d, element);
                auto& [K, F]           = system;

                constexpr array auto boundary_nodes = [] {
                    constexpr auto& boundary_table = ElementTraits< Element< T, O > >::boundary_table;
                    constexpr auto  bn_packed      = [] {
                        constexpr size_t max_nbn = std::accumulate(
                                  begin(boundary_table), end(boundary_table), 0, [](size_t val, const auto& a) {
                                return val + a.size();
                                  });
                        std::array< el_locind_t, max_nbn > ret_alloc; // NOLINT
                        for (ptrdiff_t index = 0; const auto& side_nodes : boundary_table)
                        {
                            std::ranges::copy(side_nodes, begin(ret_alloc) + index);
                            index += side_nodes.size();
                        }
                        std::ranges::sort(ret_alloc);
                        const auto [first, last] = std::ranges::unique(ret_alloc);
                        return std::make_pair(ret_alloc, std::distance(first, last));
                    }();
                    std::array< el_locind_t, bn_packed.second > ret_val; // NOLINT
                    std::copy(begin(bn_packed.first), begin(bn_packed.first) + bn_packed.second, begin(ret_val));
                    return ret_val;
                }();

                constexpr auto solution = [](const Point< 3 >& p) {
                    return p.x() + p.y() + p.z();
                };

                for (auto node : boundary_nodes)
                {
                    const auto node_location = nodePhysicalLocation(element, node);
                    const auto node_value    = solution(node_location);
                    const auto sys_index     = node * nf;
                    for (ptrdiff_t i = 0; i < static_cast< ptrdiff_t >(nf * element.n_nodes); ++i)
                    {
                        K(sys_index, i) = 0.;
                        K(i, sys_index) = 0.;
                    }
                    K(sys_index, sys_index) = 1.;
                    F[sys_index]            = node_value;
                }

                std::cout << K << '\n';

                using result_t        = typename decltype(system)::second_type;
                const result_t result = K.template selfadjointView< Eigen::Lower >().llt().solve(F);

                for (auto node : std::views::iota(0u, element.n_nodes))
                {
                    const auto node_location = nodePhysicalLocation(element, node);
                    const auto node_value    = solution(node_location);
                    const auto sys_index     = node * nf;
                    std::cout << std::setprecision(16) << result[sys_index] << '\n';
                    CHECK(result[sys_index] == Approx{node_value});
                }
            }
        };

        ho_part.cvisit(do_test, {0});
    }
}
