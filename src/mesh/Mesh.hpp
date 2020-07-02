#ifndef L3STER_INCGUARD_MESH_MESH_HPP
#define L3STER_INCGUARD_MESH_MESH_HPP

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

    Mesh(std::vector< Node< 3 > >&& nodes_, std::vector< MeshPartition >&& partitions_)
        : nodes(std::move(nodes_)), partitions(std::move(partitions_))
    {}
    Mesh(std::vector< Node< 3 > >&& nodes_, MeshPartition&& partition_) : nodes(std::move(nodes_))
    {
        partitions.emplace_back(std::move(partition_));
    }

private:
    std::vector< Node< 3 > >     nodes;      // All meshes are assumed 3D
    std::vector< MeshPartition > partitions; // Mesh = multiple partitions
};
} // namespace lstr::mesh

#endif // L3STER_INCGUARD_MESH_MESH_HPP
