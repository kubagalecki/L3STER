#ifndef L3STER_MESH_MESH_HPP
#define L3STER_MESH_MESH_HPP

#include "mesh/MeshPartition.hpp"
#include "mesh/Vertex.hpp"

#include <utility>
#include <vector>

namespace lstr
{
class Mesh
{
public:
    Mesh()            = default;
    Mesh(const Mesh&) = delete;
    Mesh(Mesh&&)      = default;
    Mesh& operator=(const Mesh&) = delete;
    Mesh& operator=(Mesh&&) = default;
    ~Mesh()                 = default;

    inline Mesh(std::vector< Vertex< 3 > >&& nodes_, MeshPartition&& partition_);
    inline Mesh(std::vector< Vertex< 3 > >&& nodes_, std::vector< MeshPartition >&& partitions_);

    [[nodiscard]] const std::vector< MeshPartition >& getPartitions() const { return partitions; }
    [[nodiscard]] std::vector< MeshPartition >&       getPartitions() { return partitions; }
    [[nodiscard]] const std::vector< Vertex< 3 > >&   getVertices() const { return vertices; }
    [[nodiscard]] std::vector< Vertex< 3 > >&         getVertices() { return vertices; }

private:
    std::vector< MeshPartition > partitions;
    std::vector< Vertex< 3 > >   vertices;
};

inline Mesh::Mesh(std::vector< Vertex< 3 > >&& nodes_, MeshPartition&& partition_) : vertices(std::move(nodes_))
{
    partitions.emplace_back(std::move(partition_));
}

inline Mesh::Mesh(std::vector< Vertex< 3 > >&& nodes_, std::vector< MeshPartition >&& partitions_)
    : partitions(std::move(partitions_)), vertices(std::move(nodes_))
{}
} // namespace lstr
#endif // L3STER_MESH_MESH_HPP
