#ifndef L3STER_MESH_MESH_HPP
#define L3STER_MESH_MESH_HPP

#include "mesh/MeshPartition.hpp"
#include "mesh/Node.hpp"

#include <utility>
#include <vector>

namespace lstr::mesh
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                         MESH CLASS                                       //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Mesh - top level interface
*/
class Mesh
{
public:
    Mesh()            = default;
    Mesh(const Mesh&) = delete;
    Mesh(Mesh&&)      = default;
    Mesh& operator=(const Mesh&) = delete;
    Mesh& operator=(Mesh&&) = default;
    ~Mesh()                 = default;

    inline Mesh(std::vector< Node< 3 > >&& nodes_, MeshPartition&& partition_);
    inline Mesh(std::vector< Node< 3 > >&& nodes_, std::vector< MeshPartition >&& partitions_);

    [[nodiscard]] const std::vector< MeshPartition >& getPartitions() const { return partitions; }
    [[nodiscard]] const std::vector< Node< 3 > >&     getNodes() const { return nodes; }
    [[nodiscard]] std::vector< MeshPartition >&       getPartitions() { return partitions; }
    [[nodiscard]] std::vector< Node< 3 > >&           getNodes() { return nodes; }

private:
    std::vector< MeshPartition > partitions; // Mesh = multiple partitions
    std::vector< Node< 3 > >     nodes;      // All meshes are assumed 3D
};

inline Mesh::Mesh(std::vector< Node< 3 > >&& nodes_, MeshPartition&& partition_)
    : nodes(std::move(nodes_))
{
    partitions.emplace_back(std::move(partition_));
}

inline Mesh::Mesh(std::vector< Node< 3 > >&& nodes_, std::vector< MeshPartition >&& partitions_)
    : partitions(std::move(partitions_)), nodes(std::move(nodes_))
{}
} // namespace lstr::mesh

#endif // L3STER_MESH_MESH_HPP
