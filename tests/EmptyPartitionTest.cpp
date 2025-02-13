#include "l3ster/algsys/MakeAlgebraicSystem.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

using namespace lstr;

template < CondensationPolicy CP >
void test()
{
    constexpr d_id_t domain_id   = 0;
    const auto       problem_def = ProblemDefinition< 1 >{{domain_id}};

    const auto     comm       = std::make_shared< MpiComm >(MPI_COMM_WORLD);
    constexpr auto mesh_order = 2;
    const auto     mesh       = generateAndDistributeMesh< mesh_order >(
        *comm, [&] { return mesh::makeSquareMesh(std::array{0., 1.}); }, {}, problem_def);

    constexpr auto alg_params       = AlgebraicSystemParams{.cond_policy = CP};
    constexpr auto algparams_ctwrpr = util::ConstexprValue< alg_params >{};
    auto           alg_sys          = makeAlgebraicSystem(comm, mesh, problem_def, {}, algparams_ctwrpr);
    alg_sys.describe();

    constexpr auto params       = KernelParams{.dimension = 2, .n_equations = 1, .n_unknowns = 1};
    constexpr auto dummy_kernel = wrapDomainEquationKernel< params >([](const auto&, auto&) {});
    alg_sys.beginAssembly();
    alg_sys.assembleProblem(dummy_kernel, std::views::single(domain_id));
    alg_sys.endAssembly();
    alg_sys.describe();
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}
