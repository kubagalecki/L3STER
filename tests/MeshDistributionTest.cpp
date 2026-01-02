#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"
#include "TestDataPath.h"

using namespace lstr;
using namespace lstr::comm;
using namespace lstr::mesh;

// Test whether the mesh is correctly distributed across the communicator
void testDistribute(const MpiComm& comm, MeshDistOpts opts)
{
    const auto mesh_path = L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh);
    const auto part      = readAndDistributeMesh< 2 >(comm, mesh_path, mesh::gmsh_tag, {}, {}, {}, opts);

    const size_t n_elems = part->getNElements();
    size_t       sum_elems{};
    comm.reduce(std::views::single(n_elems), &sum_elems, 0, MPI_SUM);
    if (comm.getRank() == 0)
    {
        constexpr size_t expected_mesh_size = 140;
        REQUIRE(sum_elems == expected_mesh_size);
    }
}

int main(int argc, char* argv[])
{
    const auto       par_scoge_guard = util::MaxParallelismGuard{4};
    L3sterScopeGuard scope_guard{argc, argv};
    MpiComm          comm{MPI_COMM_WORLD};

    testDistribute(comm, {.optimize = false});
    testDistribute(comm, {.optimize = true});
}