#ifndef L3STER_COMM_DISTRIBUTEMESH_HPP
#define L3STER_COMM_DISTRIBUTEMESH_HPP

#include "DeserializeMesh.hpp"
#include "ReceiveMesh.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"

namespace lstr
{
template < el_o_t ORDER, MeshFormat FORMAT >
MeshPartition distributeMesh(const MpiComm&               comm,
                             std::string_view             mesh_file,
                             MeshFormatTag< FORMAT >      format_tag,
                             const std::vector< d_id_t >& boundaries)
{
    constexpr int rank_zero = 0;
    const auto    n_ranks   = comm.getSize();
    const auto    my_rank   = comm.getRank();

    if (my_rank == rank_zero)
    {
        auto mesh = readMesh(mesh_file, format_tag);
        mesh.getPartitions()[0].initDualGraph();
        mesh.getPartitions()[0] = convertMeshToOrder< ORDER >(mesh.getPartitions()[0]);
        if (n_ranks == 1)
            return mesh.getPartitions()[0];
        else
        {
            mesh = partitionMesh(mesh, n_ranks, boundaries);
            for (int dest_rank = 1; dest_rank < n_ranks; ++dest_rank)
            {
                const auto serialized_part = SerializedPartition{std::move(mesh.getPartitions()[dest_rank])};
                sendPartition(comm, serialized_part, dest_rank);
            }
            return mesh.getPartitions()[0];
        }
    }
    else
        return deserializePartition(receivePartition(comm, rank_zero));
}
} // namespace lstr
#endif // L3STER_COMM_DISTRIBUTEMESH_HPP
