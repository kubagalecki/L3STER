#ifndef L3STER_COMM_DISTRIBUTEMESH_HPP
#define L3STER_COMM_DISTRIBUTEMESH_HPP

#include "DeserializeMesh.hpp"
#include "ReceiveMesh.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"

namespace lstr
{
inline MeshPartition distributeMesh(const MpiComm& comm, const Mesh& mesh, const std::vector< d_id_t >& boundaries)
{
    constexpr int rank_zero = 0;
    const auto    n_ranks   = comm.getSize();
    const auto    my_rank   = comm.getRank();
    if (my_rank == rank_zero)
    {
        if (n_ranks == 1)
            return mesh.getPartitions()[0];
        else
        {
            const auto mesh_parted = partitionMesh(mesh, n_ranks, boundaries);
            for (int dest_rank = 1; dest_rank < n_ranks; ++dest_rank)
            {
                const auto serialized_part = SerializedPartition{mesh_parted.getPartitions()[dest_rank]};
                sendPartition(comm, serialized_part, dest_rank);
            }
            return mesh_parted.getPartitions()[0];
        }
    }
    else
        return deserializePartition(receivePartition(comm, rank_zero));
}

template < el_o_t ORDER, MeshFormat FORMAT >
Mesh readAndConvertMesh(std::string_view mesh_file, MeshFormatTag< FORMAT > format_tag)
{
    auto mesh = readMesh(mesh_file, format_tag);
    mesh.getPartitions()[0].initDualGraph();
    mesh.getPartitions()[0] = convertMeshToOrder< ORDER >(mesh.getPartitions()[0]);
    return mesh;
}

template < el_o_t ORDER, MeshFormat FORMAT >
MeshPartition readAndDistributeMesh(const MpiComm&               comm,
                                    std::string_view             mesh_file,
                                    MeshFormatTag< FORMAT >      format_tag,
                                    const std::vector< d_id_t >& boundaries)
{
    const Mesh mesh = comm.getRank() == 0 ? readAndConvertMesh< ORDER >(mesh_file, format_tag) : Mesh{};
    return distributeMesh(comm, mesh, boundaries);
}
} // namespace lstr
#endif // L3STER_COMM_DISTRIBUTEMESH_HPP
