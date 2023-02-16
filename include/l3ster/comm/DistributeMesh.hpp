#ifndef L3STER_COMM_DISTRIBUTEMESH_HPP
#define L3STER_COMM_DISTRIBUTEMESH_HPP

#include "DeserializeMesh.hpp"
#include "ReceiveMesh.hpp"
#include "l3ster/comm/GatherNodeThroughputs.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"

namespace lstr
{
template < detail::ProblemDef_c auto problem_def = detail::empty_problem_def_t{} >
MeshPartition distributeMesh(const MpiComm&                comm,
                             const Mesh&                   mesh,
                             const std::vector< d_id_t >&  boundaries,
                             ConstexprValue< problem_def > probdef_ctwrpr = {})
{
    L3STER_PROFILE_FUNCTION;
    if (comm.getSize() == 1)
        return mesh.getPartitions().front();

    const auto node_throughputs = gatherNodeThroughputs(comm);
    if (comm.getRank() == 0)
    {
        const auto mesh_parted = partitionMesh(mesh, comm.getSize(), boundaries, node_throughputs, probdef_ctwrpr);
        for (int dest_rank = 1; dest_rank < comm.getSize(); ++dest_rank)
        {
            const auto serialized_part = SerializedPartition{mesh_parted.getPartitions()[dest_rank]};
            sendPartition(comm, serialized_part, dest_rank);
        }
        return mesh_parted.getPartitions().front();
    }
    else
        return deserializePartition(receivePartition(comm, 0));
}

template < el_o_t order, MeshFormat mesh_format >
Mesh readAndConvertMesh(std::string_view mesh_file, MeshFormatTag< mesh_format > format_tag)
{
    auto mesh = readMesh(mesh_file, format_tag);
    mesh.getPartitions()[0].initDualGraph();
    mesh.getPartitions()[0] = convertMeshToOrder< order >(mesh.getPartitions()[0]);
    return mesh;
}

template < el_o_t order, MeshFormat mesh_format, detail::ProblemDef_c auto problem_def = detail::empty_problem_def_t{} >
MeshPartition readAndDistributeMesh(const MpiComm&               comm,
                                    std::string_view             mesh_file,
                                    MeshFormatTag< mesh_format > format_tag,
                                    const std::vector< d_id_t >& boundaries,
                                    ConstexprValue< order >                      = {},
                                    ConstexprValue< problem_def > probdef_ctwrpr = {})
{
    const Mesh mesh = comm.getRank() == 0 ? readAndConvertMesh< order >(mesh_file, format_tag) : Mesh{};
    return distributeMesh(comm, mesh, boundaries, probdef_ctwrpr);
}
} // namespace lstr
#endif // L3STER_COMM_DISTRIBUTEMESH_HPP
