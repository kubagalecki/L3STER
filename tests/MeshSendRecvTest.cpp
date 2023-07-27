#include "l3ster/comm/ReceiveMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "TestDataPath.h"

#include <iostream>

using namespace lstr;
using namespace lstr::comm;
using namespace lstr::mesh;

int main(int argc, char* argv[])
{
    L3sterScopeGuard scope_guard{argc, argv};
    MpiComm          comm{MPI_COMM_WORLD};

    const auto read_mesh   = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), lstr::mesh::gmsh_tag);
    const auto serial_part = SerializedPartition{read_mesh};

    if (comm.getSize() <= 1)
    {
        std::cerr << "This test requires at least 2 MPI ranks";
        comm.abort();
        return EXIT_FAILURE;
    }
    for (int i = 0; i < comm.getSize(); ++i)
    {
        if (comm.getRank() == 0)
            for (int j = 1; j < comm.getSize(); ++j)
                sendPartition(comm, serial_part, j);
        else
        {
            const auto recv_part = receivePartition(comm, 0);
            try
            {
                if (recv_part.m_nodes != serial_part.m_nodes or
                    recv_part.m_n_owned_nodes != serial_part.m_n_owned_nodes)
                    throw 1;
                for (const auto& [id, dom] : recv_part.m_domains)
                {
                    const auto& dom_read = serial_part.m_domains.at(id);
                    if (dom_read.element_nodes != dom.element_nodes or dom_read.element_data != dom.element_data or
                        dom_read.element_ids != dom.element_ids or dom_read.orders != dom.orders or
                        dom_read.types != dom.types or dom_read.type_order_offsets != dom.type_order_offsets)
                        throw 1;
                }
            }
            catch (...)
            {
                std::cerr << "The received mesh is different from the sent mesh";
                comm.abort();
                return EXIT_FAILURE;
            }
        }
    }

    return EXIT_SUCCESS;
}