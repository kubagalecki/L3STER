#include "l3ster/algsys/MakeAlgebraicSystem.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

using namespace lstr;

template < OperatorEvaluationStrategy S, CondensationPolicy CP = {} >
void test()
{
    constexpr d_id_t domain_id   = 0;
    const auto       problem_def = ProblemDefinition< 1 >{{domain_id}};

    const auto     comm       = std::make_shared< MpiComm >(MPI_COMM_WORLD);
    constexpr auto mesh_order = 2;
    const auto     mesh       = generateAndDistributeMesh< mesh_order >(
        *comm, [&] { return mesh::makeSquareMesh(std::array{0., 1.}); }, {}, problem_def);

    constexpr auto alg_params       = AlgebraicSystemParams{.eval_strategy = S, .cond_policy = CP};
    constexpr auto algparams_ctwrpr = util::ConstexprValue< alg_params >{};
    auto           alg_sys          = makeAlgebraicSystem(comm, mesh, problem_def, {}, algparams_ctwrpr);
    alg_sys.describe();

    constexpr auto params       = KernelParams{.dimension = 2, .n_equations = 1, .n_unknowns = 1};
    constexpr auto dummy_kernel = wrapDomainEquationKernel< params >([](const auto&, auto&) {});
    alg_sys.beginAssembly();
    alg_sys.assembleProblem(dummy_kernel, std::views::single(domain_id));
    alg_sys.endAssembly();
    alg_sys.describe();

    // Force evaluation to actually test that everything works
    if constexpr (S == OperatorEvaluationStrategy::MatrixFree)
    {
        const auto op  = alg_sys.getOperator();
        const auto sol = alg_sys.getSolution();
        const auto rhs = alg_sys.getRhs();
        op->apply(*rhs, *sol);
    }
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< OperatorEvaluationStrategy::GlobalAssembly, CondensationPolicy::None >();
    test< OperatorEvaluationStrategy::GlobalAssembly, CondensationPolicy::ElementBoundary >();
    test< OperatorEvaluationStrategy::MatrixFree >();
}
