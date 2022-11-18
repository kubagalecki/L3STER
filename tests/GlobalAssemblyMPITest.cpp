#include "l3ster/assembly/AlgebraicSystemManager.hpp"
#include "l3ster/assembly/ComputeValuesAtNodes.hpp"
#include "l3ster/assembly/GatherGlobalValues.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

#include "Amesos2.hpp"

bool terminate_called = false;

// Solve 2D diffusion problem
int main(int argc, char* argv[])
{
    using namespace lstr;
    L3sterScopeGuard scope_guard{argc, argv};

    const MpiComm    comm;
    const std::array node_dist{0., 1., 2., 3., 4., 5., 6., 7., 8.};
    constexpr auto   mesh_order = 2;
    auto             mesh       = makeSquareMesh(node_dist);
    mesh.getPartitions()[0].initDualGraph();
    mesh.getPartitions()[0]    = convertMeshToOrder< mesh_order >(mesh.getPartitions()[0]);
    constexpr d_id_t domain_id = 0, bot_boundary = 1, top_boundary = 2, left_boundary = 3, right_boundary = 4;
    const auto my_partition = distributeMesh(comm, mesh, {bot_boundary, top_boundary, left_boundary, right_boundary});
    const auto adiabatic_boundary_ids = std::array{bot_boundary, top_boundary};
    const auto whole_boundary_ids     = std::array{top_boundary, bot_boundary, left_boundary, right_boundary};
    const auto adiabatic_bound_view   = my_partition.getBoundaryView(adiabatic_boundary_ids);
    const auto whole_bound_view       = my_partition.getBoundaryView(whole_boundary_ids);

    constexpr auto problem_def         = std::array{Pair{domain_id, std::array{true, true, true}}};
    constexpr auto dirichlet_def       = std::array{Pair{left_boundary, std::array{true, false, false}},
                                              Pair{right_boundary, std::array{true, false, false}}};
    constexpr auto probdef_ctwrpr      = ConstexprValue< problem_def >{};
    constexpr auto dirichletdef_ctwrpr = ConstexprValue< dirichlet_def >{};

    constexpr auto n_fields       = detail::deduceNFields(problem_def);
    auto           system_manager = AlgebraicSystemManager{comm, my_partition, probdef_ctwrpr, dirichletdef_ctwrpr};

    // Check that the underlying data gets cached
    {
        const auto system_manager_shallow_copy =
            AlgebraicSystemManager{comm, my_partition, probdef_ctwrpr, dirichletdef_ctwrpr};
        if (std::addressof(*system_manager.getMatrix()) != std::addressof(*system_manager_shallow_copy.getMatrix()))
        {
            std::cerr << "Algebraic system caching failure\n";
            comm.abort();
        }
    }

    // Problem assembly
    constexpr auto diffusion_kernel2d =
        [](const auto&, const std::array< std::array< val_t, 0 >, 2 >&, const SpaceTimePoint&) noexcept {
            using mat_t = Eigen::Matrix< val_t, 4, 3 >;
            std::pair< std::array< mat_t, 3 >, Eigen::Vector4d > retval;
            auto& [matrices, rhs] = retval;
            auto& [A0, A1, A2]    = matrices;
            constexpr double k    = 1.; // diffusivity

            A0       = mat_t::Zero();
            A1       = mat_t::Zero();
            A2       = mat_t::Zero();
            rhs      = Eigen::Vector4d::Zero();
            A0(1, 1) = -1.;
            A0(2, 2) = -1.;
            A1(0, 1) = k;
            A1(1, 0) = 1.;
            A1(3, 2) = 1.;
            A2(0, 2) = k;
            A2(2, 0) = 1.;
            A2(3, 1) = -1.;
            return retval;
        };
    static_assert(detail::Kernel_c< decltype(diffusion_kernel2d), 2, 0 >);
    constexpr auto neumann_bc_kernel =
        [](const auto&, const auto&, const auto&, const Eigen::Matrix< val_t, 2, 1 >& normal) noexcept {
            using mat_t = Eigen::Matrix< val_t, 1, 3 >;
            using vec_t = Eigen::Vector< val_t, 1 >;
            std::pair< std::array< mat_t, 3 >, vec_t > retval;
            auto& [matrices, rhs] = retval;
            auto& [A0, A1, A2]    = matrices;

            A0       = mat_t::Zero();
            A1       = mat_t::Zero();
            A2       = mat_t::Zero();
            rhs      = vec_t::Zero();
            A0(0, 1) = normal[0];
            A0(0, 2) = normal[1];
            return retval;
        };
    constexpr auto fieldval_getter = []< size_t N >(const std::array< n_id_t, N >& dofs) {
        return Eigen::Matrix< val_t, N, 0 >{};
    };

    constexpr auto BT       = BasisTypes::Lagrange;
    constexpr auto QT       = QuadratureTypes::GLeg;
    constexpr auto QO       = q_o_t{mesh_order * 2};
    constexpr auto dof_inds = std::array< size_t, 3 >{0, 1, 2};

    const auto assembleDomainProblem = [&] {
        system_manager.assembleDomainProblem< BT, QT, QO, dof_inds >(
            diffusion_kernel2d, my_partition, std::views::single(domain_id), fieldval_getter);
    };
    const auto assembleBoundaryProblem = [&] {
        system_manager.assembleBoundaryProblem< BT, QT, QO, dof_inds >(
            neumann_bc_kernel, adiabatic_bound_view, fieldval_getter);
    };

    // Check constraints on assembly state
    system_manager.beginAssembly();
    CHECK_THROWS(system_manager.beginModify());
    CHECK_THROWS(system_manager.endModify());
    system_manager.endAssembly();
    CHECK_THROWS(system_manager.setToZero());
    CHECK_THROWS(assembleDomainProblem());
    CHECK_THROWS(assembleBoundaryProblem());
    system_manager.beginAssembly();
    system_manager.setToZero();
    assembleDomainProblem();
    assembleBoundaryProblem();
    system_manager.endAssembly();
    system_manager.endAssembly();

    // Dirichlet BCs
    constexpr auto dirichlet_bc_val_def = [node_dist](const auto&, const auto&, const SpaceTimePoint& p) {
        Eigen::Vector< val_t, 1 > retval;
        retval[0] = p.space.x() / node_dist.back();
        return retval;
    };
    auto dirichlet_vals      = system_manager.makeSolutionMultiVector();
    auto dirichlet_vals_view = dirichlet_vals->getDataNonConst(0);
    computeValuesAtNodes(dirichlet_bc_val_def,
                         my_partition,
                         std::array{left_boundary, right_boundary},
                         system_manager.getRhsMap(),
                         ConstexprValue< std::array{0} >{},
                         empty_field_val_getter,
                         dirichlet_vals_view);

    // Check constraints on assembly state
    CHECK_THROWS(system_manager.applyDirichletBCs(*dirichlet_vals->getVector(0)));
    system_manager.beginModify();
    system_manager.beginModify();
    CHECK_THROWS(system_manager.beginAssembly());
    CHECK_THROWS(system_manager.endAssembly());
    system_manager.applyDirichletBCs(*dirichlet_vals->getVector(0));
    system_manager.endModify();
    system_manager.endModify();
    {
        auto fake_problem = AlgebraicSystemManager{comm, my_partition, probdef_ctwrpr};
        fake_problem.endAssembly();
        CHECK_THROWS(fake_problem.applyDirichletBCs(*dirichlet_vals->getVector(0)));
    }

    // Solve
    auto solution_mv = system_manager.makeSolutionMultiVector();
    auto solver      = Amesos2::KLU2< Tpetra::CrsMatrix< val_t, local_dof_t, global_dof_t >,
                                 Tpetra::MultiVector< val_t, local_dof_t, global_dof_t > >{
        system_manager.getMatrix(), solution_mv, system_manager.getRhs()};
    solver.preOrdering().symbolicFactorization().numericFactorization().solve();

    const auto     solution         = solution_mv->getVector(0);
    auto           solution_manager = SolutionManager{my_partition, comm, n_fields};
    constexpr auto field_inds       = makeIotaArray< size_t, n_fields >();
    solution_manager.updateSolution(my_partition, *solution, system_manager.getRhsMap(), field_inds, probdef_ctwrpr);
    solution_manager.communicateSharedValues();

    // Check that the underlying data cache is still usable after the solve
    {
        const auto system_manager_shallow_copy =
            AlgebraicSystemManager{comm, my_partition, probdef_ctwrpr, dirichletdef_ctwrpr};
        if (std::addressof(*system_manager.getMatrix()) != std::addressof(*system_manager_shallow_copy.getMatrix()))
        {
            std::cerr << "Algebraic system caching failed after solve\n";
            comm.abort();
        }
    }

    // Check results
    const auto     solution_view = solution->getData();
    constexpr auto compute_error =
        [node_dist](const auto& vals, const auto& ders, const SpaceTimePoint& point) noexcept {
            Eigen::Matrix< val_t, 3, 1 > error;
            const auto                   T     = vals[0];
            const auto                   dT_dx = vals[1];
            const auto                   dT_dy = vals[2];
            error[0]                           = T - point.space.x() / node_dist.back();
            error[1]                           = dT_dx - 1. / node_dist.back();
            error[2]                           = dT_dy;
            return error;
        };
    constexpr auto compute_boundary_error =
        [compute_error](const auto& vals, const auto& ders, const SpaceTimePoint& point, const auto&) noexcept {
            return compute_error(vals, ders, point);
        };
    const auto fval_getter = [&](const auto& nodes) {
        return gatherGlobalValues< field_inds >(nodes, solution_manager);
    };
    const auto error =
        computeNormL2< BT, QT, QO >(comm, compute_error, my_partition, std::views::single(domain_id), fval_getter);
    if (comm.getRank() == 0 and error.norm() > 1e-10)
    {
        std::stringstream error_msg;
        error_msg << "The error exceeded the allowed tolerance. The L2 error components were:\nvalue:\t\t\t" << error[0]
                  << "\nx derivative:\t" << error[1] << "\ny derivative:\t" << error[2] << '\n';
        std::cerr << error_msg.view();
        return EXIT_FAILURE;
    }
    const auto boundary_error =
        computeBoundaryNormL2< BT, QT, QO >(comm, compute_boundary_error, whole_bound_view, fval_getter);
    if (comm.getRank() == 0 and boundary_error.norm() > 1e-10)
    {
        std::cerr << "The error on the boundary exceeded the allowed tolerance. Since the error in the domain did not, "
                     "this is in all likelyhood an error in the computation of the norm on the boundary";
        return EXIT_FAILURE;
    }
}
