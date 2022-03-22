#include "TestDataPath.h"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/global_assembly/MakeTpetraMap.hpp"
#include "l3ster/util/GlobalResource.hpp"
#include "l3ster/util/KokkosScopeGuard.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;
    GlobalResource< MpiScopeGuard >::initialize(argc, argv);
    GlobalResource< KokkosScopeGuard >::initialize(argc, argv);
    MpiComm comm{};

    const auto     mesh = distributeMesh< 2 >(comm, L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), gmsh_tag, {});
    constexpr auto problem_def   = ConstexprValue< [] {
        constexpr std::array cov{false, true};
        return std::array{Pair{d_id_t{1}, cov}};
    }() >{};
    const auto     dof_intervals = computeDofIntervals(mesh, problem_def, comm);
    const auto     tpetra_map    = makeTpetraMap(mesh.getNodes(), dof_intervals, comm);
    const auto     map_entries   = tpetra_map->getNodeElementList();
    if (std::ranges::equal(mesh.getNodes(), tpetra_map->getNodeElementList()))
        return EXIT_SUCCESS;
    else
        return EXIT_FAILURE;
}