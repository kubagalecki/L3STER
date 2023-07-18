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
    constexpr d_id_t domain_id      = 0;
    constexpr auto   problem_def    = std::array{util::Pair{domain_id, std::array{true}}};
    constexpr auto   probdef_ctwrpr = util::ConstexprValue< problem_def >{};

    const auto     comm       = MpiComm{MPI_COMM_WORLD};
    constexpr auto mesh_order = 2;
    const auto     mesh       = generateAndDistributeMesh< mesh_order >(
        comm,
        [&] {
            return mesh::makeSquareMesh(std::array{0., 1.});
        },
        {1, 2, 3, 4},
        {},
        probdef_ctwrpr);

    auto alg_sys = makeAlgebraicSystem(comm, mesh, CondensationPolicyTag< CP >{}, probdef_ctwrpr);
    alg_sys->describe(comm);
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}
