#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"
#include "TestDataPath.h"

using namespace lstr;
using namespace lstr::comm;
using namespace lstr::mesh;

using el_ptr_var = Domain< 1 >::el_univec_t::const_ptr_variant_t;
bool compareElemsAt(el_ptr_var v1, el_ptr_var v2)
{
    constexpr auto compare_contents = [](auto ptr1, auto ptr2) {
        if constexpr (std::same_as< decltype(ptr1), decltype(ptr2) >)
            return *ptr1 == *ptr2;
        return false;
    };
    return std::visit(compare_contents, v1, v2);
}

// Test whether the mesh is sent and received correctly
void testSendRecv(const MpiComm& comm)
{
    if (comm.getSize() == 1)
        return;

    const auto read_mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), {}, lstr::mesh::gmsh_tag);
    if (comm.getRank() == 0)
    {
        auto reqs = std::vector< MpiComm::Request >{};
        for (int i = 1; i < comm.getSize(); ++i)
            sendMesh(comm, read_mesh, i, std::back_inserter(reqs));
        MpiComm::Request::waitAll(reqs);
    }
    else
    {
        const auto recv_mesh = receiveMesh< 1 >(comm, 0);
        REQUIRE(std::ranges::equal(read_mesh.getNodeOwnership().owned(), recv_mesh.getNodeOwnership().owned()));
        REQUIRE(std::ranges::equal(read_mesh.getNodeOwnership().shared(), recv_mesh.getNodeOwnership().shared()));
        REQUIRE(std::ranges::equal(read_mesh.getDomainIds(), recv_mesh.getDomainIds()));
        for (d_id_t domain_id : read_mesh.getDomainIds())
        {
            const auto& read_domain = read_mesh.getDomain(domain_id);
            const auto& recv_domain = recv_mesh.getDomain(domain_id);
            REQUIRE(read_domain.dim == recv_domain.dim);
            REQUIRE(read_domain.elements.size() == recv_domain.elements.size());
            for (size_t i = 0; i != read_domain.elements.size(); ++i)
                REQUIRE(compareElemsAt(read_domain.elements.at(i), recv_domain.elements.at(i)));
        }
    }

    comm.barrier();
}

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

    testSendRecv(comm);
    testDistribute(comm, {.optimize = false});
    testDistribute(comm, {.optimize = true});
}