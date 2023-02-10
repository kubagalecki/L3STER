#define L3STER_ELEMENT_ORDERS 4
#include "Amesos2.hpp"
#include "DataPath.h"

#include "l3ster/l3ster.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;
    using namespace std::string_view_literals;

    L3sterScopeGuard scope_guard{argc, argv};
    const MpiComm    comm;

    constexpr auto        mesh_order   = L3STER_ELEMENT_ORDERS;
    constexpr d_id_t      domain_id    = 0;
    static constexpr auto boundary_ids = makeIotaArray< d_id_t, 6 >(1);

    constexpr auto node_dist = std::invoke([] {
        constexpr size_t                   edge_divs = 12;
        constexpr auto                     dx        = 1. / static_cast< val_t >(edge_divs);
        std::array< val_t, edge_divs + 1 > retval{};
        for (double x = 0; auto& r : retval)
        {
            r = x;
            x += dx;
        }
        return retval;
    });

    Mesh mesh;
    if (comm.getRank() == 0)
    {
        mesh = makeCubeMesh(node_dist);
        mesh.getPartitions()[0].initDualGraph();
        mesh.getPartitions()[0] = convertMeshToOrder< mesh_order >(mesh.getPartitions()[0]);
    }
    const auto my_partition =
        distributeMesh(comm, mesh, std::vector< d_id_t >(boundary_ids.begin(), boundary_ids.end()));
    const auto boundary_view = my_partition.getBoundaryView(boundary_ids);

    {
        std::stringstream log_msg;
        log_msg << "Rank " << comm.getRank() << "\n\tNumber of elements: " << my_partition.getNElements()
                << "\n\tNumber of owned nodes: " << my_partition.getOwnedNodes().size()
                << "\n\tNumber of ghost nodes: " << my_partition.getGhostNodes().size() << '\n';
        std::cout << log_msg.view();
    }

    constexpr auto problem_def         = std::array{Pair{domain_id, std::array{true, true, true, true}}};
    constexpr auto dirichlet_def       = std::invoke([] {
        std::array< Pair< d_id_t, std::array< bool, 4 > >, boundary_ids.size() > retval{};
        std::ranges::transform(boundary_ids, retval.begin(), [](auto d) {
            return Pair{d, std::array{true, false, false, false}};
        });
        return retval;
    });
    constexpr auto probdef_ctwrpr      = ConstexprValue< problem_def >{};
    constexpr auto dirichletdef_ctwrpr = ConstexprValue< dirichlet_def >{};

    constexpr auto n_fields    = detail::deduceNFields(problem_def);
    constexpr auto field_inds  = makeIotaArray< size_t, n_fields >();
    constexpr auto T_inds      = std::array< size_t, 1 >{0};
    constexpr auto T_grad_inds = std::array< size_t, 3 >{1, 2, 3};
    const auto field_inds_view = std::array{std::span< const size_t >{T_inds}, std::span< const size_t >{T_grad_inds}};
    constexpr auto dof_inds    = field_inds;
    constexpr auto BT          = BasisTypes::Lagrange;
    constexpr auto QT          = QuadratureTypes::GLeg;
    constexpr auto QO          = q_o_t{mesh_order * 2};

    constexpr auto diffusion_kernel3d =
        []< typename T >(const auto&, const std::array< T, 3 >&, const SpaceTimePoint&) noexcept {
            using mat_t = Eigen::Matrix< val_t, 7, 4 >;
            using vec_t = Eigen::Vector< double, 7 >;
            std::pair< std::array< mat_t, 4 >, vec_t > retval;
            auto& [matrices, rhs]  = retval;
            auto& [A0, Ax, Ay, Az] = matrices;

            constexpr double k = 1.; // diffusivity
            constexpr double s = 1.; // source

            A0  = mat_t::Zero();
            Ax  = mat_t::Zero();
            Ay  = mat_t::Zero();
            Az  = mat_t::Zero();
            rhs = vec_t::Zero();

            // -k * div T = s
            Ax(0, 1) = -k;
            Ay(0, 2) = -k;
            Az(0, 3) = -k;
            rhs[0]   = s;

            // grad T = q
            A0(1, 1) = -1.;
            Ax(1, 0) = 1.;
            A0(2, 2) = -1.;
            Ay(2, 0) = 1.;
            A0(3, 3) = -1.;
            Az(3, 0) = 1.;

            // rot q = 0
            Ay(4, 3) = 1.;
            Az(4, 2) = -1.;
            Ax(5, 3) = -1.;
            Az(5, 1) = 1.;
            Ax(6, 2) = 1.;
            Ay(6, 1) = -1.;

            return retval;
        };

    auto system_manager   = makeAlgebraicSystemManager(comm, my_partition, probdef_ctwrpr, dirichletdef_ctwrpr);
    auto solution_manager = SolutionManager{my_partition, comm, n_fields};
    system_manager->beginAssembly();
    system_manager->assembleDomainProblem< BT, QT, QO, dof_inds >(
        diffusion_kernel3d, my_partition, std::views::single(domain_id), empty_field_val_getter);
    system_manager->endAssembly();

    constexpr auto dirichlet_bc_val_def = [](const auto&, const auto&, const SpaceTimePoint& p) {
        Eigen::Vector< val_t, 1 > retval;
        retval[0] = 0.;
        return retval;
    };

    auto dirichlet_vals = system_manager->getDirichletBCValueVector()->getLocalViewHost(Tpetra::Access::ReadWrite);
    computeValuesAtNodes(dirichlet_bc_val_def,
                         my_partition,
                         boundary_ids,
                         system_manager->getDofMap(),
                         ConstexprValue< T_inds >{},
                         empty_field_val_getter,
                         asSpan(Kokkos::subview(dirichlet_vals, Kokkos::ALL, 0)));

    system_manager->beginModify();
    system_manager->applyDirichletBCs();
    system_manager->endModify();
}