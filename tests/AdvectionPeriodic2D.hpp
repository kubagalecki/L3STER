#include "l3ster/algsys/MakeAlgebraicSystem.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/solve/BelosSolvers.hpp"
#include "l3ster/solve/NativePreconditioners.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

using namespace lstr;
using namespace lstr::mesh;

const auto node_dist_x = util::linspaceVector(-.5, .5, 5);
const auto node_dist_y = util::linspaceVector(0., .5, 4);
const auto W           = node_dist_x.back() - node_dist_x.front();
const auto H           = node_dist_y.back() - node_dist_y.front();

template < el_o_t mesh_order = 1 >
auto makeMesh2D(const MpiComm& comm, auto probdef_ctwrpr, const std::vector< double >& nd = node_dist_x)
{
    return generateAndDistributeMesh< mesh_order >(comm, [&] { return makeSquareMesh(nd); }, {}, probdef_ctwrpr);
}

template < OperatorEvaluationStrategy S, CondensationPolicy CP >
void solveAdvection2D()
{
    constexpr d_id_t domain_id = 0, bot_bound = 1, top_bound = 2, left_bound = 3, right_bound = 4;
    constexpr auto   boundary_ids   = std::array{bot_bound, top_bound, left_bound, right_bound};
    constexpr auto   problem_def    = ProblemDef{defineDomain< 1 >(domain_id, ALL_DOFS)};
    constexpr auto   probdef_ctwrpr = util::ConstexprValue< problem_def >{};
    auto             bc_def         = BCDefinition< problem_def.n_fields >{};
    bc_def.definePeriodic({left_bound}, {right_bound}, {node_dist_x.back() - node_dist_x.front(), 0., 0.});
    bc_def.defineDirichlet({top_bound, bot_bound});

    constexpr auto mesh_order = 4;
    const auto     comm       = std::make_shared< MpiComm >(MPI_COMM_WORLD);
    const auto     mesh       = generateAndDistributeMesh< mesh_order >(
        *comm, [&] { return makeSquareMesh(node_dist_x, node_dist_y); }, {}, probdef_ctwrpr);
    summarizeMesh(*comm, *mesh);

    constexpr auto alg_params       = AlgebraicSystemParams{.eval_strategy = S, .cond_policy = CP};
    constexpr auto algparams_ctwrpr = util::ConstexprValue< alg_params >{};
    auto           alg_sys          = makeAlgebraicSystem(comm, mesh, probdef_ctwrpr, bc_def, algparams_ctwrpr);
    alg_sys.describe();

    //    // BDF 2
    //    constexpr auto time_order       = 2;
    //    constexpr auto bdf_leading      = 1.5;
    //    constexpr auto bdf_coefs        = std::array< double, time_order >{2., -.5};

    // BDF 3
    constexpr auto time_order  = 3;
    constexpr auto bdf_leading = 11. / 6.;
    constexpr auto bdf_coefs   = std::array< double, time_order >{3., -1.5, 1. / 3.};

    constexpr double u = 1., v = 0.;
    constexpr double dt       = .05;
    constexpr auto adv_params = KernelParams{.dimension = 2, .n_equations = 1, .n_unknowns = 1, .n_fields = time_order};
    const auto     advection_kernel2d = wrapDomainEquationKernel< adv_params >([&](const auto& in, auto& out) {
        const auto& [field_vals, field_ders, point] = in;
        auto& [operators, rhs]                      = out;
        auto& [A0, A1, A2]                          = operators;
        A0(0, 0)                                    = bdf_leading;
        A1(0, 0)                                    = u * dt;
        A2(0, 0)                                    = v * dt;
        rhs(0, 0) = std::transform_reduce(field_vals.begin(), field_vals.end(), bdf_coefs.begin(), 0.); // inner product
    });

    constexpr auto anal_sol = [](const auto& in, auto& out) {
        const auto t    = in.point.time;
        const auto x    = in.point.space.x();
        const auto dv   = t * u;
        auto       x_dv = x - dv;
        while (x_dv < node_dist_x.front())
            x_dv += node_dist_x.back() - node_dist_x.front();

        out[0] = std::exp(-10. * x_dv * x_dv);
    };
    constexpr auto solution_params    = KernelParams{.dimension = 2, .n_equations = 1};
    constexpr auto solution_kernel    = wrapDomainResidualKernel< solution_params >(anal_sol);
    constexpr auto solution_kernel_bc = wrapBoundaryResidualKernel< solution_params >(anal_sol);
    constexpr auto i0                 = std::array{0};

    auto time_hist_inds   = util::makeIotaArray< size_t, time_order >();
    auto solution_manager = SolutionManager{*mesh, time_order};
    for (auto i : time_hist_inds)
    {
        const auto time = static_cast< double >(i) * -dt;
        solution_manager.setFields(*comm, *mesh, solution_kernel, {domain_id}, {i}, {}, time);
        solution_manager.setFields(*comm, *mesh, solution_kernel_bc, boundary_ids, {i}, {}, time);
    }

    constexpr auto precond_opts = NativeJacobiOpts{};
    auto           solver       = CG{{}, precond_opts};
    const auto     num_steps    = std::lround((node_dist_x.back() - node_dist_x.front()) / dt);
    for (long time_step = 1; time_step <= num_steps; ++time_step)
    {
        const auto time = static_cast< double >(time_step) * dt;
        alg_sys.setDirichletBCValues(solution_kernel_bc, {bot_bound, top_bound}, i0, {}, time);
        alg_sys.beginAssembly();
        alg_sys.assembleProblem(advection_kernel2d, {domain_id}, solution_manager.makeFieldValueGetter(time_hist_inds));
        alg_sys.endAssembly();
        alg_sys.solve(solver);
        const auto last_ind = time_hist_inds.back();
        alg_sys.updateSolution(i0, solution_manager, std::views::single(last_ind));
        std::ranges::rotate(time_hist_inds, std::prev(time_hist_inds.end()));
    }

    constexpr auto error_params = KernelParams{.dimension = 2, .n_equations = 1, .n_fields = 1};
    const auto     error_kernel = wrapDomainResidualKernel< error_params >([&](const auto& in, auto& out) {
        const auto computed = in.field_vals[0];
        anal_sol(in, out);
        out[0] = computed - out[0];
    });
    const auto     error        = computeNormL2(*comm,
                                     error_kernel,
                                     *mesh,
                                     std::views::single(domain_id),
                                     solution_manager.makeFieldValueGetter(std::array{time_hist_inds.front()}),
                                                {},
                                     dt * static_cast< double >(num_steps))[0] /
                       (W * H) * 100.;
    constexpr auto thresh = 5.;
    if (comm->getRank() == 0)
        std::cout << std::format("Normalized L2 error: {:.2f}%  ->  {}", error, error < thresh ? "PASS" : "FAIL")
                  << std::endl;
    REQUIRE(error < thresh);
}