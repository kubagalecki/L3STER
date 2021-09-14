#include "comm/DistributeMesh.hpp"
#include "util/GlobalResource.hpp"

#include "TestDataPath.h"

#include <iostream>

int main(int argc, char* argv[])
{
    using namespace lstr;
    GlobalResource< MpiScopeGuard >::init(argc, argv);
    MpiComm comm{};

    const auto   part    = distributeMesh< 2 >(comm, L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), gmsh_tag, {});
    const size_t n_elems = part.getNElements();

    size_t sum_elems{};
    comm.reduce(&n_elems, &sum_elems, 1, 0, MPI_SUM);

    if (comm.getRank() == 0)
    {
        constexpr size_t expected_mesh_size = 140;
        if (sum_elems == expected_mesh_size)
            return EXIT_SUCCESS;
        else
        {
            std::cerr << "Distributed partitions' sizes don't sum up to the total mesh size.\nExpected sum:  "
                      << expected_mesh_size << "\nComputed sum: " << sum_elems << '\n';
            comm.abort();
            return EXIT_FAILURE;
        }
    }
}
