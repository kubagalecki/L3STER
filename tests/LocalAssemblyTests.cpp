#include "catch2/catch.hpp"

#include "l3ster/local_assembly/AssembleLocalSystem.hpp"
#include "l3ster/mapping/ComputeBasisDerivative.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"
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
                CHECK(der(basis, Eigen::all).sum() == Approx{0.}.margin(1e-13));
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

TEST_CASE("Local system assembly tests", "[local_asm]")
{
    // Solve problems using a 1 element discretization and compare with known results

    SECTION("Diffusion 2D")
    {
        constexpr auto        ET       = ElementTypes::Quad;
        constexpr el_o_t      EO       = 4;
        constexpr auto        QT       = QuadratureTypes::GLeg;
        constexpr q_o_t       QO       = 2 * EO;
        constexpr auto        BT       = BasisTypes::Lagrange;
        constexpr auto        el_nodes = typename Element< ET, EO >::node_array_t{};
        ElementData< ET, EO > data{{Point{1., -1., 0.}, Point{2., -1., 0.}, Point{1., 1., 1.}, Point{2., 1., 1.}}};
        const auto            element = Element< ET, EO >{el_nodes, data, 0};

        constexpr size_t nf                  = 3;
        constexpr size_t ne                  = 4;
        constexpr size_t dim                 = 2;
        constexpr auto   diffusion_kernel_2d = []() noexcept {
            using A_t   = Eigen::Matrix< val_t, ne, nf >;
            using F_t   = Eigen::Matrix< val_t, ne, 1 >;
            using ret_t = std::pair< std::array< A_t, dim + 1 >, F_t >;
            ret_t ret_val;
            auto& [A0, A1, A2] = ret_val.first;
            auto& F            = ret_val.second;
            for (auto& mat : ret_val.first)
                mat.setZero();
            F.setZero();

            constexpr double lambda = 1.;

            A0(1, 1) = -1.;
            A0(2, 2) = -1.;

            A1(0, 1) = lambda;
            A1(1, 0) = 1.;
            A1(3, 2) = 1.;

            A2(0, 2) = lambda;
            A2(2, 0) = 1.;
            A2(3, 1) = -1.;

            return ret_val;
        };

        constexpr auto solution = [](const Point< 3 >& p) {
            return p.x();
        };

        auto system  = assembleLocalSystem< QT, QO, BT >(diffusion_kernel_2d, element);
        auto& [K, F] = system;
        K            = K.template selfadjointView< Eigen::Lower >();
        auto u       = F;

        constexpr auto boundary_nodes = [] {
            constexpr auto& boundary_table = ElementTraits< Element< ET, EO > >::boundary_table;
            constexpr auto  bn_packed      = [] {
                constexpr size_t max_nbn =
                    std::accumulate(begin(boundary_table), end(boundary_table), 0, [](size_t val, const auto& a) {
                        return val + a.size();
                          });
                std::array< el_locind_t, max_nbn > ret_alloc{};
                for (size_t index = 0; const auto& side_nodes : boundary_table)
                {
                    std::ranges::copy(side_nodes, begin(ret_alloc) + index);
                    index += side_nodes.size();
                }
                std::ranges::sort(ret_alloc);
                const auto [first, last] = std::ranges::unique(ret_alloc);
                return std::make_pair(ret_alloc, std::distance(begin(ret_alloc), first));
            }();
            std::array< el_locind_t, bn_packed.second > ret_val{};
            std::copy(begin(bn_packed.first), begin(bn_packed.first) + bn_packed.second, begin(ret_val));
            return ret_val;
        }();
        REQUIRE(boundary_nodes.size() == (EO + 1) * (EO + 1) - (EO - 1) * (EO - 1));

        constexpr auto bc_inds = [&] {
            auto ret_val = boundary_nodes;
            for (auto& bc_ind : ret_val)
                bc_ind *= nf;
            return ret_val;
        }();
        constexpr auto nonbc_inds = [&] {
            std::array< ptrdiff_t, Element< ET, EO >::n_nodes * nf - bc_inds.size() > ret_val{};
            std::ranges::set_difference(std::views::iota(0u, Element< ET, EO >::n_nodes * nf), bc_inds, begin(ret_val));
            return ret_val;
        }();
        Eigen::Matrix< val_t, bc_inds.size(), 1 > bc_vals{};
        for (ptrdiff_t i = 0; auto node : boundary_nodes)
            bc_vals[i++] = solution(nodePhysicalLocation(element, node));

        Eigen::Matrix< val_t, nonbc_inds.size(), nonbc_inds.size() > K_red = K(nonbc_inds, nonbc_inds);
        Eigen::Matrix< val_t, nonbc_inds.size(), 1 > F_red = F(nonbc_inds, 1) - K(nonbc_inds, bc_inds) * bc_vals;
        auto                                         u_red = F_red;

        u_red         = K_red.llt().solve(F_red);
        u(nonbc_inds) = u_red;
        u(bc_inds)    = bc_vals;

        for (auto node : std::views::iota(0u, element.n_nodes))
        {
            const auto node_location = nodePhysicalLocation(element, node);
            const auto node_value    = solution(node_location);
            const auto sys_index     = node * nf;
            CHECK(u[sys_index] == Approx{node_value});
        }
    }

    // TODO: write a few more tests, including 3D ones; blocked by: Eigen issue #2375
}
