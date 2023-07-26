#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/glob_asm/AlgebraicSystem.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/solve/Solvers.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

using namespace lstr;

template < CondensationPolicy CP >
void test()
{
    const auto comm = MpiComm{MPI_COMM_WORLD};

    constexpr d_id_t domain_id = 0, bot_boundary = 1, top_boundary = 2, left_boundary = 3, right_boundary = 4;
    constexpr auto   problem_def    = ProblemDef{defineDomain< 3 >(domain_id, 0, 1, 2)};
    constexpr auto   probdef_ctwrpr = util::ConstexprValue< problem_def >{};
    constexpr auto   dirichlet_def =
        ProblemDef{defineDomain< 3 >(left_boundary, 0), defineDomain< 3 >(right_boundary, 0)};
    constexpr auto dirichletdef_ctwrpr = util::ConstexprValue< dirichlet_def >{};

    constexpr auto node_dist  = std::array{0., 1., 2., 3., 4., 5., 6.};
    constexpr auto mesh_order = 2;
    const auto     mesh       = generateAndDistributeMesh< mesh_order >(
        comm,
        [&] { return mesh::makeSquareMesh(node_dist); },
        {bot_boundary, top_boundary, left_boundary, right_boundary},
        {},
        probdef_ctwrpr);
    const auto adiabatic_bound_view = mesh.getBoundaryView(std::array{bot_boundary, top_boundary});
    const auto whole_bound_view =
        mesh.getBoundaryView(std::array{top_boundary, bot_boundary, left_boundary, right_boundary});

    auto alg_sys = makeAlgebraicSystem(comm, mesh, CondensationPolicyTag< CP >{}, probdef_ctwrpr, dirichletdef_ctwrpr);

    // Check that the underlying data gets cached
    {
        const auto system_manager_shallow_copy =
            makeAlgebraicSystem(comm, mesh, CondensationPolicyTag< CP >{}, probdef_ctwrpr, dirichletdef_ctwrpr);
        REQUIRE(alg_sys.get() == system_manager_shallow_copy.get());
    }

    constexpr auto diffusion_kernel2d =
        [](const auto&, const std::array< std::array< val_t, 0 >, 2 >&, const SpaceTimePoint&) noexcept {
            auto retval           = std::pair< std::array< Eigen::Matrix< val_t, 4, 3 >, 3 >, Eigen::Vector4d >{};
            auto& [matrices, rhs] = retval;
            auto& [A0, A1, A2]    = matrices;
            constexpr double k    = 1.; // diffusivity
            A0.setZero();
            A1.setZero();
            A2.setZero();
            rhs.setZero();
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
    static_assert(glob_asm::Kernel_c< decltype(diffusion_kernel2d), 2, 0 >);
    constexpr auto neumann_bc_kernel =
        [](const auto&, const auto&, const auto&, const Eigen::Matrix< val_t, 2, 1 >& normal) noexcept {
            auto retval = std::pair< std::array< Eigen::Matrix< val_t, 1, 3 >, 3 >, Eigen::Vector< val_t, 1 > >{};
            auto& [matrices, rhs] = retval;
            auto& [A0, A1, A2]    = matrices;
            A0.setZero();
            A1.setZero();
            A2.setZero();
            rhs.setZero();
            A0(0, 1) = normal[0];
            A0(0, 2) = normal[1];
            return retval;
        };
    constexpr auto dirichlet_bc_val_def = [node_dist](const auto&, const auto&, const SpaceTimePoint& p) {
        auto retval = Eigen::Vector< val_t, 1 >{};
        retval[0]   = p.space.x() / node_dist.back();
        return retval;
    };

    const auto assembleDomainProblem = [&] {
        alg_sys->assembleDomainProblem(diffusion_kernel2d, mesh, std::views::single(domain_id));
    };
    const auto assembleBoundaryProblem = [&] {
        alg_sys->assembleBoundaryProblem(neumann_bc_kernel, adiabatic_bound_view);
    };

    // Check constraints on assembly state
    alg_sys->beginAssembly();
    assembleDomainProblem();
    assembleBoundaryProblem();
    CHECK_THROWS(alg_sys->applyDirichletBCs());
    alg_sys->endAssembly(mesh);
    alg_sys->describe(comm);
    CHECK_THROWS(assembleDomainProblem());
    CHECK_THROWS(assembleBoundaryProblem());
    CHECK_THROWS(alg_sys->endAssembly(mesh));

    alg_sys->setDirichletBCValues(
        dirichlet_bc_val_def, mesh, std::array{left_boundary, right_boundary}, util::ConstexprValue< std::array{0} >{});
    alg_sys->applyDirichletBCs();

    {
        auto fake_problem = makeAlgebraicSystem(comm, mesh, CondensationPolicyTag< CP >{}, probdef_ctwrpr);
        fake_problem->endAssembly(mesh);
        CHECK_THROWS(fake_problem->applyDirichletBCs());
    }

    constexpr auto dof_inds = util::makeIotaArray< size_t, 3 >();

    auto solver   = solvers::Lapack{};
    auto solution = alg_sys->makeSolutionVector();
    alg_sys->solve(solver, solution);
    auto solution_manager = SolutionManager{mesh, problem_def.n_fields};
    alg_sys->updateSolution(mesh, solution, dof_inds, solution_manager, dof_inds);

    // Check that the underlying data cache is still usable after the solve
    {
        const auto system_manager_shallow_copy =
            makeAlgebraicSystem(comm, mesh, CondensationPolicyTag< CP >{}, probdef_ctwrpr, dirichletdef_ctwrpr);
        REQUIRE(alg_sys.get() == system_manager_shallow_copy.get());
    }

    // Check results
    constexpr auto compute_error =
        [node_dist](const auto& vals, [[maybe_unused]] const auto& ders, const SpaceTimePoint& point) noexcept {
            auto       error = Eigen::Matrix< val_t, 3, 1 >{};
            const auto T     = vals[0];
            const auto dT_dx = vals[1];
            const auto dT_dy = vals[2];
            error[0]         = T - point.space.x() / node_dist.back();
            error[1]         = dT_dx - 1. / node_dist.back();
            error[2]         = dT_dy;
            return error;
        };
    constexpr auto compute_boundary_error =
        [compute_error](const auto& vals, const auto& ders, const SpaceTimePoint& point, const auto&) noexcept {
            return compute_error(vals, ders, point);
        };
    const auto fval_getter = solution_manager.makeFieldValueGetter(dof_inds);

    const auto error          = computeNormL2(comm, compute_error, mesh, std::views::single(domain_id), fval_getter);
    const auto boundary_error = computeBoundaryNormL2(comm, compute_boundary_error, whole_bound_view, fval_getter);
    if (comm.getRank() == 0 and error.norm() >= 1e-10)
    {
        std::stringstream error_msg;
        error_msg << "The error exceeded the allowed tolerance. The L2 error components were:\nvalue:\t\t\t" << error[0]
                  << "\nx derivative:\t" << error[1] << "\ny derivative:\t" << error[2] << '\n';
        std::cerr << error_msg.view();
    }
    comm.barrier();
    REQUIRE(error.norm() < 1e-10);
    REQUIRE(boundary_error.norm() < 1e-10);

    alg_sys->beginAssembly();
}

// Solve 2D diffusion problem
int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}
