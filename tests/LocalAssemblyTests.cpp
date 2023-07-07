#include "catch2/catch.hpp"

#include "l3ster/assembly/AssembleLocalSystem.hpp"
#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"
#include "l3ster/mapping/ComputePhysBasisDer.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"

using namespace lstr;

TEST_CASE("Local system assembly", "[local_asm]")
{
    // Solve problems using a 1 element discretization and compare with known results

    SECTION("Diffusion 2D")
    {
        constexpr auto              ET       = mesh::ElementType::Quad;
        constexpr el_o_t            EO       = 4;
        constexpr auto              QT       = quad::QuadratureType::GaussLegendre;
        constexpr q_o_t             QO       = 11; // Needs to be large enough for the local system buffer to overflow
        constexpr auto              BT       = basis::BasisType::Lagrange;
        constexpr auto              el_nodes = typename mesh::Element< ET, EO >::node_array_t{};
        mesh::ElementData< ET, EO > data{
            {mesh::Point{1., -1., 0.}, mesh::Point{2., -1., 0.}, mesh::Point{1., 1., 1.}, mesh::Point{2., 1., 1.}}};
        const auto element = mesh::Element< ET, EO >{el_nodes, data, 0};

        const auto& basis_at_q = basis::getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();

        constexpr size_t nf                  = 3;
        constexpr size_t ne                  = 4;
        constexpr size_t dim                 = 2;
        constexpr auto   diffusion_kernel_2d = [](const auto&, const auto&, const auto&) noexcept {
            using A_t   = Eigen::Matrix< val_t, ne, nf >;
            using F_t   = Eigen::Matrix< val_t, ne, 1 >;
            using ret_t = std::pair< std::array< A_t, dim + 1 >, F_t >;
            ret_t retval;
            auto& [A0, A1, A2] = retval.first;
            auto& F            = retval.second;
            for (auto& mat : retval.first)
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

            return retval;
        };

        constexpr auto solution = [](const mesh::Point< 3 >& p) {
            return p.x();
        };

        const auto& [K, F] = assembleLocalSystem(
            diffusion_kernel_2d, element, Eigen::Matrix< val_t, element.n_nodes, 0 >{}, basis_at_q, 0.);
        auto u = F;

        constexpr auto boundary_nodes = std::invoke([] {
            constexpr auto& boundary_table = mesh::ElementTraits< mesh::Element< ET, EO > >::boundary_table;
            constexpr auto  bn_packed      = std::invoke([] {
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
            });
            std::array< el_locind_t, bn_packed.second > retval{};
            std::copy(begin(bn_packed.first), begin(bn_packed.first) + bn_packed.second, begin(retval));
            return retval;
        });
        REQUIRE(boundary_nodes.size() == (EO + 1) * (EO + 1) - (EO - 1) * (EO - 1));

        constexpr auto                            bc_inds    = std::invoke([&] {
            auto retval = boundary_nodes;
            for (auto& bc_ind : retval)
                bc_ind *= nf;
            return retval;
        });
        constexpr auto                            nonbc_inds = std::invoke([&] {
            std::array< ptrdiff_t, mesh::Element< ET, EO >::n_nodes * nf - bc_inds.size() > retval{};
            std::ranges::set_difference(
                std::views::iota(0u, mesh::Element< ET, EO >::n_nodes * nf), bc_inds, begin(retval));
            return retval;
        });
        Eigen::Matrix< val_t, bc_inds.size(), 1 > bc_vals{};
        for (ptrdiff_t i = 0; auto node : boundary_nodes)
            bc_vals[i++] = solution(mesh::nodePhysicalLocation(element, node));

        Eigen::Matrix< val_t, nonbc_inds.size(), nonbc_inds.size() > K_red = K(nonbc_inds, nonbc_inds);
        Eigen::Matrix< val_t, nonbc_inds.size(), 1 > F_red = F(nonbc_inds, 1) - K(nonbc_inds, bc_inds) * bc_vals;
        auto                                         u_red = F_red;

        u_red         = K_red.llt().solve(F_red);
        u(nonbc_inds) = u_red;
        u(bc_inds)    = bc_vals;

        for (el_locind_t node = 0; node != element.n_nodes; ++node)
        {
            const auto node_location = nodePhysicalLocation(element, node);
            const auto node_value    = solution(node_location);
            const auto sys_index     = node * nf;
            CHECK(u[sys_index] == Approx{node_value});
        }
    }
}
