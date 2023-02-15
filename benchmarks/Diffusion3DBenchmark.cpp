#include "BelosSolverFactory_Tpetra.hpp"
#include "Ifpack2_Factory.hpp"

#define L3STER_ELEMENT_ORDERS 4
#include "l3ster/l3ster.hpp"

#include "DataPath.h"

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
    constexpr auto field_names = std::array{"T"sv, "gradT"sv};
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

            // -k * div q = s
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
    constexpr auto error_kernel =
        []< typename DerT >(const auto& vals, const std::array< DerT, 3 >& ders, const SpaceTimePoint& point) noexcept {
            Eigen::Matrix< val_t, 4, 1 > error;
            const auto& [T, qx, qy, qz]          = vals;
            const auto& [x_ders, y_ders, z_ders] = ders;

            const auto& q_xx = x_ders[1];
            const auto& q_yy = y_ders[2];
            const auto& q_zz = z_ders[3];
            const auto& T_x  = x_ders[0];
            const auto& T_y  = y_ders[0];
            const auto& T_z  = z_ders[0];

            constexpr double k = 1.;
            constexpr double s = 1.;

            error[0] = k * (q_xx + q_yy + q_zz) + s;
            error[1] = T_x - qx;
            error[2] = T_y - qy;
            error[3] = T_z - qz;
            return error;
        };

    auto system_manager = makeAlgebraicSystemManager(comm, my_partition, probdef_ctwrpr, dirichletdef_ctwrpr);
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

    L3STER_PROFILE_REGION_BEGIN("Set up Ifpack2 Chebyshev preconditioner");
    auto precond_params = makeTeuchosRCP< Teuchos::ParameterList >();
    precond_params->set("chebyshev: degree", 3);
    Ifpack2::Factory precond_factory;
    auto             preconditioner = precond_factory.create("CHEBYSHEV", system_manager->getMatrix());
    preconditioner->setParameters(*precond_params);
    L3STER_PROFILE_REGION_BEGIN("Initialize");
    preconditioner->initialize();
    L3STER_PROFILE_REGION_END("Initialize");
    L3STER_PROFILE_REGION_BEGIN("Compute");
    preconditioner->compute();
    L3STER_PROFILE_REGION_END("Compute");
    L3STER_PROFILE_REGION_END("Set up Ifpack2 Chebyshev preconditioner");

    L3STER_PROFILE_REGION_BEGIN("Set up Belos::LinearProblem");
    auto algebraic_problem = makeTeuchosRCP< Belos::LinearProblem< val_t, tpetra_multivector_t, tpetra_operator_t > >(
        system_manager->getMatrix(), system_manager->getSolutionVector(), system_manager->getRhs());
    algebraic_problem->setLeftPrec(preconditioner);
    if (not algebraic_problem->setProblem())
        throw std::runtime_error{"Failed to set up Belos::LinearProblem"};
    L3STER_PROFILE_REGION_END("Set up Belos::LinearProblem");

    L3STER_PROFILE_REGION_BEGIN("Set up Belos::BlockCGSolMgr");
    auto solver_params = makeTeuchosRCP< Teuchos::ParameterList >();
    solver_params->set("Block Size", 1);
    solver_params->set("Maximum Iterations", 10'000);
    solver_params->set("Convergence Tolerance", 1.e-6);
    solver_params->set("Verbosity",
                       Belos::Warnings + Belos::IterationDetails + Belos::FinalSummary + Belos::TimingDetails);
    solver_params->set("Output Frequency", 100);
    Belos::SolverFactory< val_t, tpetra_multivector_t, tpetra_operator_t > solver_factory;
    auto solver = solver_factory.create("Block CG", solver_params);
    solver->setProblem(algebraic_problem);
    L3STER_PROFILE_REGION_END("Set up Belos::BlockCGSolMgr");

    L3STER_PROFILE_REGION_BEGIN("Solve the algebraic problem");
    if (solver->solve() != Belos::Converged)
        throw std::runtime_error{"Solver failed to converge"};
    L3STER_PROFILE_REGION_END("Solve the algebraic problem");

    L3STER_PROFILE_REGION_BEGIN("Solution management");
    auto solution_manager = SolutionManager{my_partition, comm, n_fields};
    solution_manager.updateSolution(
        my_partition, *system_manager->getSolutionVector(), system_manager->getDofMap(), field_inds, probdef_ctwrpr);
    solution_manager.communicateSharedValues();
    L3STER_PROFILE_REGION_END("Solution management");

    L3STER_PROFILE_REGION_BEGIN("Compute solution error");
    const auto fval_getter = [&](const auto& nodes) {
        return gatherGlobalValues< field_inds >(nodes, solution_manager);
    };
    const auto error =
        computeNormL2< BT, QT, QO >(comm, error_kernel, my_partition, std::views::single(domain_id), fval_getter);
    L3STER_PROFILE_REGION_END("Compute solution error");

    if (comm.getRank() == 0)
    {
        std::cout << "The L2 error components are:";
        for (int i = 0; i < error.size(); ++i)
            std::cout << "\n\t" << error[i];
        std::cout << std::endl;
    }

    L3STER_PROFILE_REGION_BEGIN("Export results to VTK");
    auto exporter = PvtuExporter{my_partition, solution_manager.getNodeMap()};
    exporter.exportSolution("Cube_Diffusion.pvtu", comm, solution_manager, field_names, field_inds_view);
    L3STER_PROFILE_REGION_END("Export results to VTK");
}