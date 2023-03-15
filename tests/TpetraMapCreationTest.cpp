#include "Common.hpp"
#include "TestDataPath.h"
#include "l3ster/assembly/MakeTpetraMap.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;
    L3sterScopeGuard scope_guard{argc, argv};
    MpiComm          comm{MPI_COMM_WORLD};

    Mesh mesh_full;
    if (comm.getRank() == 0)
    {
        constexpr std::array dist{0., 1., 2., 3., 4.};
        constexpr auto       order = 2;
        mesh_full                  = makeCubeMesh(dist);
        mesh_full.getPartitions()[0].initDualGraph();
        mesh_full.getPartitions()[0] = convertMeshToOrder< order >(mesh_full.getPartitions()[0]);
    }
    const auto     mesh           = distributeMesh(comm, mesh_full, {});
    constexpr auto probdef_ctwrpr = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{false, true}}} >{};
    const auto     cond_map = detail::NodeCondensationMap::makeBoundaryNodeCondensationMap(comm, mesh, probdef_ctwrpr);
    auto           owned_condensed = detail::getCondensedOwnedNodesView(mesh, cond_map);
    const auto     dof_intervals   = computeDofIntervals(comm, mesh, cond_map, probdef_ctwrpr);
    const auto     tpetra_map      = makeTpetraMap(owned_condensed, dof_intervals, comm);
    const auto     map_entries     = tpetra_map->getLocalElementList();
    REQUIRE(std::ranges::equal(owned_condensed, tpetra_map->getLocalElementList()));
    REQUIRE(tpetra_map->isOneToOne());
}