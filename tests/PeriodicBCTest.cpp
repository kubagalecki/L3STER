#include "l3ster/bcs/PeriodicBC.hpp"
#include "l3ster/algsys/MakeAlgebraicSystem.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/VtkExport.hpp"
#include "l3ster/solve/Amesos2Solvers.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

using namespace lstr;
using namespace lstr::mesh;
using namespace lstr::bcs;

constexpr auto node_dist = util::linspaceArray< 5 >(-1., 1.);
constexpr auto N         = node_dist.size();
constexpr auto dx        = N - 1;
constexpr auto dy        = dx * N;
constexpr auto dz        = dy * N;

template < el_o_t mesh_order = 1 >
auto makeMesh2D(const MpiComm& comm, auto probdef_ctwrpr)
{
    return generateAndDistributeMesh< mesh_order >(comm, [&] { return makeSquareMesh(node_dist); }, {}, probdef_ctwrpr);
}

template < el_o_t mesh_order = 1 >
auto makeMesh3D(const MpiComm& comm, auto probdef_ctwrpr)
{
    return generateAndDistributeMesh< mesh_order >(comm, [&] { return makeCubeMesh(node_dist); }, {}, probdef_ctwrpr);
}

template < size_t dim >
void assertCorrectNumEdges(const MpiComm& comm, const bcs::PeriodicBC< dim >& bc)
{
    auto dir_to_edge_local = std::array< size_t, dim >{};
    for (auto&& [from, to] : bc)
        for (size_t i = 0; i != dim; ++i)
            if (to[i] != invalid_node)
                ++dir_to_edge_local[i];
    auto dir_to_edge_global = std::array< size_t, dim >{};
    comm.allReduce(dir_to_edge_local, dir_to_edge_global.begin(), MPI_SUM);
    static constexpr auto boundary_size = dim == 2 ? node_dist.size() : node_dist.size() * node_dist.size();
    REQUIRE(std::ranges::all_of(dir_to_edge_global, [](size_t e) { return e == boundary_size; }));
}

void testBoundaryMatching2D()
{
    constexpr d_id_t domain_id = 0, bot_bound = 1, top_bound = 2, left_bound = 3, right_bound = 4;
    constexpr auto   problem_def    = ProblemDef{defineDomain< 2 >(domain_id, ALL_DOFS)};
    constexpr auto   probdef_ctwrpr = util::ConstexprValue< problem_def >{};
    const auto       comm           = std::make_shared< MpiComm >(MPI_COMM_WORLD);
    const auto       mesh           = makeMesh2D(*comm, probdef_ctwrpr);
    auto             period_def     = PeriodicBCDefinition< 2 >{};
    period_def.definePeriodicBoundary({bot_bound}, {top_bound}, {0., node_dist.back() - node_dist.front(), 0.}, {0});
    period_def.definePeriodicBoundary({left_bound}, {right_bound}, {node_dist.back() - node_dist.front(), 0., 0.}, {1});
    const auto bc = PeriodicBC{period_def, *mesh, *comm};
    assertCorrectNumEdges(*comm, bc);
    if (comm->getSize() == 1) // Nodes get renumbered during partitioning, so we only know the result for comm size = 1
    {
        for (n_id_t n = 0; n != N; ++n)
            REQUIRE(bc.lookup(n)[0] == n + dy or bc.lookup(n + dy)[0] == n);
        for (n_id_t n = 0; n < N * N; n += N)
            REQUIRE(bc.lookup(n)[1] == n + dx or bc.lookup(n + dx)[1] == n);
    }
}

void testBoundaryMatching3D()
{
    constexpr d_id_t domain_id = 0;
    constexpr d_id_t bot_bound = 1, top_bound = 2, left_bound = 3, right_bound = 4, back_bound = 5, front_bound = 6;
    constexpr auto   problem_def    = ProblemDef{defineDomain< 3 >(domain_id, ALL_DOFS)};
    constexpr auto   probdef_ctwrpr = util::ConstexprValue< problem_def >{};
    const auto       comm           = std::make_shared< MpiComm >(MPI_COMM_WORLD);
    const auto       mesh           = makeMesh3D(*comm, probdef_ctwrpr);
    auto             period_def     = PeriodicBCDefinition< 3 >{};
    period_def.definePeriodicBoundary({bot_bound}, {top_bound}, {0., 0., node_dist.back() - node_dist.front()}, {0});
    period_def.definePeriodicBoundary({left_bound}, {right_bound}, {0., node_dist.back() - node_dist.front(), 0.}, {1});
    period_def.definePeriodicBoundary({back_bound}, {front_bound}, {node_dist.back() - node_dist.front(), 0., 0.}, {2});
    const auto bc = PeriodicBC{period_def, *mesh, *comm};
    assertCorrectNumEdges(*comm, bc);
    if (comm->getSize() == 1) // Nodes get renumbered during partitioning, so we only know the result for comm size = 1
    {
        for (n_id_t x = 0; x != N; ++x)
            for (n_id_t y = 0; y != N; ++y)
            {
                const auto n1 = y * node_dist.size() + x;
                const auto n2 = n1 + dz;
                REQUIRE(bc.lookup(n1)[0] == n2 or bc.lookup(n2)[0] == n1);
            }
        for (n_id_t x = 0; x != N; ++x)
            for (n_id_t z = 0; z != N; ++z)
            {
                const auto n1 = z * N * N + x;
                const auto n2 = n1 + dy;
                REQUIRE(bc.lookup(n1)[1] == n2 or bc.lookup(n2)[1] == n1);
            }
        for (n_id_t y = 0; y != N; ++y)
            for (n_id_t z = 0; z != N; ++z)
            {
                const auto n1 = z * N * N + y * N;
                const auto n2 = n1 + dx;
                REQUIRE(bc.lookup(n1)[2] == n2 or bc.lookup(n2)[2] == n1);
            }
    }
}

template < OperatorEvaluationStrategy S, CondensationPolicy CP >
void solveAdvection2D()
{
    [[maybe_unused]] constexpr d_id_t domain_id = 0, bot_bound = 1, top_bound = 2, left_bound = 3, right_bound = 4;
    constexpr auto                    problem_def    = ProblemDef{defineDomain< 1 >(domain_id, ALL_DOFS)};
    constexpr auto                    probdef_ctwrpr = util::ConstexprValue< problem_def >{};
    auto                              bc_def         = BCDefinition< problem_def.n_fields >{};
    // bc_def.definePeriodic({left_bound}, {right_bound}, {node_dist.back() - node_dist.front(), 0., 0.});
    bc_def.defineDirichlet({top_bound, bot_bound});

    const auto comm = std::make_shared< MpiComm >(MPI_COMM_WORLD);
    const auto mesh = makeMesh2D< 4 >(*comm, probdef_ctwrpr);
    summarizeMesh(*comm, *mesh);

    constexpr auto alg_params       = AlgebraicSystemParams{.eval_strategy = S, .cond_policy = CP};
    constexpr auto algparams_ctwrpr = util::ConstexprValue< alg_params >{};
    auto           alg_sys          = makeAlgebraicSystem(comm, mesh, probdef_ctwrpr, bc_def, algparams_ctwrpr);

    constexpr double u = 1., v = 0.;
    constexpr double dt         = .1;
    constexpr auto   adv_params = KernelParams{.dimension = 2, .n_equations = 1, .n_unknowns = 1, .n_fields = 1};
    constexpr auto   advection_kernel2d = wrapDomainEquationKernel< adv_params >([](const auto& in, auto& out) {
        const auto& [field_vals, field_ders, point] = in;
        const auto phi_prev                         = field_vals[0];
        auto& [operators, rhs]                      = out;
        auto& [A0, A1, A2]                          = operators;
        A0(0, 0)                                    = 1.;
        A1(0, 0)                                    = u * dt;
        A2(0, 0)                                    = v * dt;
        rhs(0, 0)                                   = phi_prev;
    });

    constexpr auto anal_sol = [](const auto& in, auto& out) {
        const auto t    = in.point.time;
        const auto x    = in.point.space.x();
        const auto dv   = t * u;
        auto       x_dv = x - dv;
        while (x_dv < node_dist.front())
            x_dv += node_dist.back() - node_dist.front();

        out[0] = std::exp(-16. * x_dv * x_dv);
    };
    constexpr auto solution_params    = KernelParams{.dimension = 2, .n_equations = 1};
    constexpr auto solution_kernel    = wrapDomainResidualKernel< solution_params >(anal_sol);
    constexpr auto solution_kernel_bc = wrapBoundaryResidualKernel< solution_params >(anal_sol);

    auto solution_manager = SolutionManager{*mesh, 1};
    alg_sys.endAssembly();
    alg_sys.setValues(alg_sys.getSolution(), solution_kernel, {domain_id}, {0}, {}, 0.);
    constexpr auto inds = std::array{0};
    alg_sys.updateSolution(inds, solution_manager, inds);
    auto exporter = PvtuExporter{comm, *mesh};
    exporter.exportSolution("results/phi_00.pvtu", solution_manager, {"phi"}, std::views::single(inds));

    auto solver = solvers::KLU2{};
    for (int time_step = 1; time_step <= 20; ++time_step)
    {
        const auto time = time_step * dt;
        alg_sys.setDirichletBCValues(solution_kernel_bc, {bot_bound, top_bound}, inds, {}, time);
        alg_sys.beginAssembly();
        alg_sys.assembleProblem(advection_kernel2d, {domain_id}, solution_manager.makeFieldValueGetter(inds));
        alg_sys.endAssembly();
        alg_sys.solve(solver);
        alg_sys.updateSolution(inds, solution_manager, inds);
        const auto file_name = std::format("results/phi_{:02}.pvtu", time_step);
        exporter.exportSolution(file_name, solution_manager, {"phi"}, std::views::single(inds));
    }
}

// Solve 2D diffusion problem
int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{1};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    testBoundaryMatching2D();
    testBoundaryMatching3D();
    solveAdvection2D< OperatorEvaluationStrategy::GlobalAssembly, CondensationPolicy::None >();
}
