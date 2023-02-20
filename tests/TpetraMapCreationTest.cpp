#include "TestDataPath.h"
#include "l3ster/assembly/MakeTpetraMap.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/GlobalResource.hpp"
#include "l3ster/util/ScopeGuards.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;
    GlobalResource< MpiScopeGuard >::initialize(argc, argv);
    GlobalResource< KokkosScopeGuard >::initialize(argc, argv);
    MpiComm comm{MPI_COMM_WORLD};

    Mesh mesh_full;
    if (comm.getRank() == 0)
    {
        constexpr std::array dist{0., 1., 2., 3., 4.};
        constexpr auto       order = 2;
        mesh_full                  = makeCubeMesh(dist);
        mesh_full.getPartitions()[0].initDualGraph();
        mesh_full.getPartitions()[0] = convertMeshToOrder< order >(mesh_full.getPartitions()[0]);
    }
    const auto mesh = distributeMesh(comm, mesh_full, {});

    constexpr auto problem_def   = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{false, true}}} >{};
    const auto     dof_intervals = computeDofIntervals(mesh, problem_def, comm);
    const auto     tpetra_map    = makeTpetraMap(mesh.getOwnedNodes(), dof_intervals, comm);
    const auto     map_entries   = tpetra_map->getLocalElementList();
    if (std::ranges::equal(mesh.getOwnedNodes(), tpetra_map->getLocalElementList()))
        return EXIT_SUCCESS;
    else
        return EXIT_FAILURE;
}