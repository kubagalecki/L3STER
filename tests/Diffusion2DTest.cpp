#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/glob_asm/AlgebraicSystem.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/solve/Amesos2Solvers.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

using namespace lstr;
using namespace lstr::glob_asm;
using namespace lstr::mesh;

constexpr auto node_dist = std::array{0., 1., 2., 3., 4., 5., 6.};

auto makeMesh(const MpiComm& comm, auto probdef_ctwrpr)
{
    constexpr auto mesh_order = 2;
    return generateAndDistributeMesh< mesh_order >(comm, [&] { return makeSquareMesh(node_dist); }, {}, probdef_ctwrpr);
}

template < CondensationPolicy CP >
void test()
{
    const auto comm = std::make_shared< MpiComm >(MPI_COMM_WORLD);

    constexpr d_id_t domain_id = 0, bot_boundary = 1, top_boundary = 2, left_boundary = 3, right_boundary = 4;
    constexpr auto   problem_def    = ProblemDef{defineDomain< 3 >(domain_id, ALL_DOFS)};
    constexpr auto   probdef_ctwrpr = util::ConstexprValue< problem_def >{};
    constexpr auto   dirichlet_def =
        ProblemDef{defineDomain< 3 >(left_boundary, 0), defineDomain< 3 >(right_boundary, 0)};
    constexpr auto dirichletdef_ctwrpr = util::ConstexprValue< dirichlet_def >{};

    const auto mesh = makeMesh(*comm, probdef_ctwrpr);

    constexpr auto adiabatic_bound_ids = std::array{bot_boundary, top_boundary};
    constexpr auto boundary_ids        = std::array{top_boundary, bot_boundary, left_boundary, right_boundary};

    constexpr auto alg_params       = AlgebraicSystemParams{.cond_policy = CP};
    constexpr auto algparams_ctwrpr = util::ConstexprValue< alg_params >{};
    auto           alg_sys = makeAlgebraicSystem(comm, mesh, probdef_ctwrpr, dirichletdef_ctwrpr, algparams_ctwrpr);

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

    const auto assembleDomainProblem = [&] {
        alg_sys.assembleProblem(diffusion_kernel2d, std::views::single(domain_id));
    };
    const auto assembleBoundaryProblem = [&] {
        alg_sys.assembleProblem(neumann_bc_kernel, adiabatic_bound_ids);
    };

    constexpr auto dirichlet_bound_ids = std::array{left_boundary, right_boundary};
    alg_sys.setDirichletBCValues(dirichlet_bc_kernel, dirichlet_bound_ids, std::array{0});

    // Check constraints on assembly state
    alg_sys.beginAssembly();
    assembleDomainProblem();
    assembleBoundaryProblem();
    alg_sys.endAssembly();
    alg_sys.describe();
    CHECK_THROWS(assembleDomainProblem());
    CHECK_THROWS(assembleBoundaryProblem());
    CHECK_THROWS(alg_sys.endAssembly());

    {
        auto dummy_problem = makeAlgebraicSystem(comm, mesh, probdef_ctwrpr, {}, algparams_ctwrpr);
        dummy_problem.endAssembly();
    }

    constexpr auto dof_inds = util::makeIotaArray< size_t, 3 >();

    auto solver   = solvers::Lapack{};
    auto solution = alg_sys.initSolution();
    alg_sys.solve(solver, solution);
    auto solution_manager = SolutionManager{*mesh, problem_def.n_fields};
    alg_sys.updateSolution(solution, dof_inds, solution_manager, dof_inds);

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
    if (comm->getRank() == 0 and error.norm() >= 1e-10)
    {
        std::stringstream error_msg;
        error_msg << "The error exceeded the allowed tolerance. The L2 error components were:\nvalue:\t\t\t" << error[0]
                  << "\nx derivative:\t" << error[1] << "\ny derivative:\t" << error[2] << '\n';
        std::cerr << error_msg.view();
    }
    comm->barrier();
    REQUIRE(error.norm() < 1e-10);
    REQUIRE(boundary_error.norm() < 1e-10);

    alg_sys.beginAssembly();
}

// Solve 2D diffusion problem
int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}
