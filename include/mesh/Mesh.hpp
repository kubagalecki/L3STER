#ifndef L3STER_MESH_MESH_HPP
#define L3STER_MESH_MESH_HPP

#include "mesh/MeshPartition.hpp"
#include "mesh/Point.hpp"

#include <utility>
#include <vector>

namespace lstr
{
class Mesh
{
public:
    Mesh(const Mesh&) = delete;
    Mesh(Mesh&&)      = default;
    Mesh& operator=(const Mesh&) = delete;
    Mesh& operator=(Mesh&&) = default;
    ~Mesh()                 = default;

    inline Mesh(std::vector< MeshPartition > partitions_) : partitions{std::move(partitions_)} {}

    [[nodiscard]] const std::vector< MeshPartition >& getPartitions() const { return partitions; }
    [[nodiscard]] std::vector< MeshPartition >&       getPartitions() { return partitions; }

private:
    std::vector< MeshPartition > partitions;
};
} // namespace lstr
#endif // L3STER_MESH_MESH_HPP
