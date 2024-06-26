#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/glob_asm/AlgebraicSystem.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

using namespace lstr;

template < CondensationPolicy CP >
void test()
{
    constexpr d_id_t domain_id      = 0;
    constexpr auto   problem_def    = ProblemDef{defineDomain< 1 >(domain_id, 0)};
    constexpr auto   probdef_ctwrpr = util::ConstexprValue< problem_def >{};

    const auto     comm       = std::make_shared< MpiComm >(MPI_COMM_WORLD);
    constexpr auto mesh_order = 2;
    const auto     mesh       = generateAndDistributeMesh< mesh_order >(
        *comm, [&] { return mesh::makeSquareMesh(std::array{0., 1.}); }, {}, probdef_ctwrpr);

    constexpr auto alg_params       = AlgebraicSystemParams{.cond_policy = CP};
    constexpr auto algparams_ctwrpr = util::ConstexprValue< alg_params >{};
    auto           alg_sys          = makeAlgebraicSystem(comm, mesh, probdef_ctwrpr, {}, algparams_ctwrpr);
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
