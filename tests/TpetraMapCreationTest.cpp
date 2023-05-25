#include "Common.hpp"
#include "TestDataPath.h"
#include "l3ster/assembly/MakeTpetraMap.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

using namespace lstr;

template < CondensationPolicy CP >
void test()
{
    const auto     comm            = MpiComm{MPI_COMM_WORLD};
    constexpr auto order           = 2;
    const auto     mesh            = generateAndDistributeMesh< order >(comm,
                                                         [] {
                                                             return makeCubeMesh(std::array{0., 1., 2., 3., 4.});
                                                         },
                                                         {});
    constexpr auto probdef_ctwrpr  = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{false, true}}} >{};
    const auto     cond_map        = detail::makeCondensationMap< CP >(comm, mesh, probdef_ctwrpr);
    auto           owned_condensed = detail::getCondensedOwnedNodesView(mesh, cond_map);
    const auto     dof_intervals   = computeDofIntervals(comm, mesh, cond_map, probdef_ctwrpr);
    const auto     tpetra_map      = makeTpetraMap(owned_condensed, dof_intervals, comm);
    const auto     map_entries     = tpetra_map->getLocalElementList();
    REQUIRE(std::ranges::equal(owned_condensed, tpetra_map->getLocalElementList()));
    REQUIRE(tpetra_map->isOneToOne());
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = detail::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}