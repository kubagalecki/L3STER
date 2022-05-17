#ifndef L3STER_MESH_MESH_HPP
#define L3STER_MESH_MESH_HPP

#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/mesh/Point.hpp"

namespace lstr
{
class Mesh
{
public:
    Mesh() = default;
    explicit Mesh(std::vector< MeshPartition > partitions_) : partitions{std::move(partitions_)} {}

    [[nodiscard]] const std::vector< MeshPartition >& getPartitions() const { return partitions; }
    [[nodiscard]] std::vector< MeshPartition >&       getPartitions() { return partitions; }

private:
    std::vector< MeshPartition > partitions;
};
} // namespace lstr
#endif // L3STER_MESH_MESH_HPP
