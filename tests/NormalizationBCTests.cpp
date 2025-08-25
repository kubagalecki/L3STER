#include "l3ster/algsys/MakeAlgebraicSystem.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/LineMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/solve/BelosSolvers.hpp"
#include "l3ster/solve/NativePreconditioners.hpp"

#include "Common.hpp"
#include "Kernels.hpp"

using namespace lstr;
using namespace lstr::algsys;
using namespace lstr::mesh;

constexpr auto node_dist = util::linspaceArray< 5 >(0., 1.);

auto makeMesh(const MpiComm& comm, const auto& problem_def)
{
    constexpr auto mesh_order = 2;
    return generateAndDistributeMesh< mesh_order >(comm, [&] { return makeLineMesh(node_dist); }, {}, problem_def);
}

template < CondensationPolicy CP, OperatorEvaluationStrategy S, LocalEvalStrategy LEV = LocalEvalStrategy::Auto >
void test()
{
    const auto comm = std::make_shared< MpiComm >(MPI_COMM_WORLD);

    constexpr d_id_t domain_id   = 0;
    const auto       problem_def = ProblemDefinition< 3 >{{domain_id}};
    auto             bc_def      = BCDefinition{problem_def};
    bc_def.normalize({0, 2});

    const auto mesh = makeMesh(*comm, problem_def);
    summarizeMesh(*comm, *mesh);

    constexpr auto alg_params       = AlgebraicSystemParams{.eval_strategy = S, .cond_policy = CP};
    constexpr auto algparams_ctwrpr = util::ConstexprValue< alg_params >{};
    auto           alg_sys          = makeAlgebraicSystem(comm, mesh, problem_def, bc_def, algparams_ctwrpr);

    constexpr auto params   = KernelParams{.dimension = 1, .n_equations = 3, .n_unknowns = 3};
    constexpr auto equation = [](const auto& in, auto& out) {
        auto& [operators, rhs] = out;
        auto& [A0, Ax]         = operators;

        Ax(0, 0) = 1.;
        rhs[0]   = 1.;

        A0(1, 1) = 1.;
        rhs[1]   = 1.;

        Ax(2, 2) = 1.;
        rhs[2]   = 2 * in.point.space.x();
    };
    constexpr auto kernel = wrapDomainEquationKernel< params >(equation);

    constexpr auto asm_opts        = AssemblyOptions{.value_order = 1, .derivative_order = 0, .eval_strategy = LEV};
    constexpr auto asm_opts_ctwrpr = util::ConstexprValue< asm_opts >{};

    // Check constraints on assembly state
    alg_sys.beginAssembly();
    alg_sys.assembleProblem(kernel, std::views::single(domain_id), {}, {}, asm_opts_ctwrpr);
    alg_sys.endAssembly();
    alg_sys.describe();

    constexpr auto dof_inds = util::makeIotaArray< size_t, 3 >();

    constexpr auto solver_opts  = IterSolverOpts{.tol = 1e-10};
    constexpr auto precond_opts = NativeJacobiOpts{};
    auto           solver       = CG{solver_opts, precond_opts};
    alg_sys.solve(solver);
    auto solution_manager = SolutionManager{*mesh, 3};
    alg_sys.updateSolution(dof_inds, solution_manager, dof_inds);

    // Check results
    constexpr auto error = [](const auto& in, auto& out) {
        const auto& vals = in.field_vals;
        out[0]           = vals[0] - in.point.space.x();
        out[1]           = vals[1] - 1.;
        out[2]           = vals[2] - in.point.space.x() * in.point.space.x();
    };
    constexpr auto error_params = KernelParams{.dimension = 1, .n_equations = 3, .n_fields = 3};
    constexpr auto error_kernel = wrapDomainResidualKernel< error_params >(error);
    const auto     field_access = solution_manager.getFieldAccess(dof_inds);

    const auto error_vec = computeNormL2< asm_opts >(*comm, error_kernel, *mesh, {domain_id}, field_access);
    if (comm->getRank() == 0)
        std::println("L2 error components:\n{:.3e}\n{:.3e}\n{:.3e}", error_vec[0], error_vec[1], error_vec[2]);
    comm->barrier();
    constexpr auto eps = 1e-8;
    REQUIRE(error_vec.norm() < eps);
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None, OperatorEvaluationStrategy::GlobalAssembly >();
    test< CondensationPolicy::ElementBoundary, OperatorEvaluationStrategy::GlobalAssembly >();
    test< CondensationPolicy::None, OperatorEvaluationStrategy::MatrixFree >();
}
