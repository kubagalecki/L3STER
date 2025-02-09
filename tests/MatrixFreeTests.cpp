
#include "Common.hpp"

#include "l3ster/algsys/MakeAlgebraicSystem.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/solve/BelosSolvers.hpp"
#include "l3ster/solve/NativePreconditioners.hpp"
#include "l3ster/util/ScopeGuards.hpp"

using namespace lstr;
using namespace lstr::algsys;
using namespace lstr::mesh;

constexpr auto node_dist = util::linspaceArray< 5 >(0., 1.);

auto makeMesh(const MpiComm& comm, auto probdef_ctwrpr)
{
    constexpr auto mesh_order = 2;
    return generateAndDistributeMesh< mesh_order >(comm, [&] { return makeSquareMesh(node_dist); }, {}, probdef_ctwrpr);
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{3};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    const auto comm          = std::make_shared< MpiComm >(MPI_COMM_WORLD);

    constexpr d_id_t domain_id = 0, bot_boundary = 1, top_boundary = 2, left_boundary = 3, right_boundary = 4;
    constexpr auto   problem_def    = ProblemDef{defineDomain< 3 >(domain_id, ALL_DOFS)};
    constexpr auto   probdef_ctwrpr = util::ConstexprValue< problem_def >{};
    auto             bc_def         = BCDefinition< problem_def.n_fields >{};
    bc_def.defineDirichlet({left_boundary, right_boundary}, {0});

    const auto mesh = makeMesh(*comm, probdef_ctwrpr);
    summarizeMesh(*comm, *mesh);

    constexpr auto adiabatic_bound_ids = std::array{bot_boundary, top_boundary};
    constexpr auto boundary_ids        = std::array{top_boundary, bot_boundary, left_boundary, right_boundary};

    constexpr auto alg_params       = AlgebraicSystemParams{.eval_strategy = OperatorEvaluationStrategy::MatrixFree};
    constexpr auto algparams_ctwrpr = util::ConstexprValue< alg_params >{};
    auto           alg_sys          = makeAlgebraicSystem(comm, mesh, probdef_ctwrpr, bc_def, algparams_ctwrpr);
    alg_sys.describe();

    constexpr auto diff_params = KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = 3};
    constexpr auto diffusion_kernel2d =
        wrapDomainEquationKernel< diff_params >([]([[maybe_unused]] const auto& in, auto& out) {
            auto& [operators, rhs] = out;
            auto& [A0, A1, A2]     = operators;
            constexpr double k     = 1.; // diffusivity
            A0(1, 1)               = -1.;
            A0(2, 2)               = -1.;
            A1(0, 1)               = k;
            A1(1, 0)               = 1.;
            A1(3, 2)               = 1.;
            A2(0, 2)               = k;
            A2(2, 0)               = 1.;
            A2(3, 1)               = -1.;
        });
    constexpr auto neubc_params      = KernelParams{.dimension = 2, .n_equations = 1, .n_unknowns = 3};
    constexpr auto neumann_bc_kernel = wrapBoundaryEquationKernel< neubc_params >([](const auto& in, auto& out) {
        const auto& [vals, ders, point, normal] = in;
        auto& [operators, rhs]                  = out;
        auto& [A0, A1, A2]                      = operators;
        A0(0, 1)                                = normal[0];
        A0(0, 2)                                = normal[1];
    });

    constexpr auto dirbc_params        = KernelParams{.dimension = 2, .n_equations = 1};
    constexpr auto dirichlet_bc_kernel = wrapBoundaryResidualKernel< dirbc_params >(
        [](const auto& in, auto& out) { out[0] = in.point.space.x() / node_dist.back(); });
    constexpr auto dirichlet_bound_ids = std::array{left_boundary, right_boundary};
    alg_sys.setDirichletBCValues(dirichlet_bc_kernel, dirichlet_bound_ids, std::array{0});

    // Check constraints on assembly state
    alg_sys.beginAssembly();
    alg_sys.assembleProblem(diffusion_kernel2d, std::views::single(domain_id));
    alg_sys.assembleProblem(neumann_bc_kernel, adiabatic_bound_ids);
    alg_sys.endAssembly();

    constexpr auto solver_opts  = IterSolverOpts{.tol = 1e-10};
    constexpr auto precond_opts = NativeJacobiOpts{};
    auto           solver       = CG{solver_opts, precond_opts};
    alg_sys.solve(solver);

    constexpr auto dof_inds         = util::makeIotaArray< size_t, 3 >();
    auto           solution_manager = SolutionManager{*mesh, diff_params.n_unknowns};
    alg_sys.updateSolution(dof_inds, solution_manager, dof_inds);

    // Check results
    constexpr auto params        = KernelParams{.dimension = 2, .n_equations = 3, .n_fields = 3};
    constexpr auto compute_error = [](const auto& in, auto& error) {
        const auto& point             = in.point;
        const auto& vals              = in.field_vals;
        const auto& [T, dT_dx, dT_dy] = vals;
        error[0]                      = T - point.space.x() / node_dist.back();
        error[1]                      = dT_dx - 1. / node_dist.back();
        error[2]                      = dT_dy;
    };
    constexpr auto dom_error_kernel = wrapDomainResidualKernel< params >(compute_error);
    constexpr auto bnd_error_kernel = wrapBoundaryResidualKernel< params >(compute_error);
    const auto     fval_getter      = solution_manager.makeFieldValueGetter(dof_inds);

    const auto error = computeNormL2(*comm, dom_error_kernel, *mesh, std::views::single(domain_id), fval_getter);
    const auto boundary_error = computeNormL2(*comm, bnd_error_kernel, *mesh, boundary_ids, fval_getter);
    if (comm->getRank() == 0)
        std::cout << std::format("L2 error components:\n{:<15}{}\n{:<15}{}\n{:<15}{}\n",
                                 "value:",
                                 error[0],
                                 "x derivative:",
                                 error[1],
                                 "y derivative:",
                                 error[2]);
    comm->barrier();
    constexpr auto eps = 1.e-8;
    REQUIRE(error.norm() < eps);
    REQUIRE(boundary_error.norm() < eps);

    alg_sys.beginAssembly(); // for code coverage
}