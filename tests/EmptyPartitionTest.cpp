#include "l3ster/assembly/AlgebraicSystem.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
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
    constexpr auto   problem_def    = std::array{Pair{domain_id, std::array{true}}};
    constexpr auto   probdef_ctwrpr = ConstexprValue< problem_def >{};

    const auto comm = MpiComm{MPI_COMM_WORLD};
    const auto mesh = std::invoke([&] {
        if (comm.getRank() == 0)
        {
            constexpr el_o_t mesh_order = 2;
            auto             full_mesh  = makeSquareMesh(std::vector{0., 1.});
            full_mesh.getPartitions().front().initDualGraph();
            full_mesh.getPartitions().front() = convertMeshToOrder< mesh_order >(full_mesh.getPartitions().front());
            return distributeMesh(comm, full_mesh, {1, 2, 3, 4}, probdef_ctwrpr);
        }
        else
            return distributeMesh(comm, {}, {}, probdef_ctwrpr);
    });

    auto alg_sys = makeAlgebraicSystem(comm, mesh, CondensationPolicyTag< CP >{}, probdef_ctwrpr);
    alg_sys->describe(comm);
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = detail::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}
