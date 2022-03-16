#ifndef L3STER_ASSEMBLY_NODELOCALGLOBALCONVERTER_HPP
#define L3STER_ASSEMBLY_NODELOCALGLOBALCONVERTER_HPP

#include "l3ster/mesh/Mesh.hpp"

// We are relying on the STL's hashmap, since the performance of converting node IDs between local and global values is
// not critical. This should only happen during setup, not during solution.
#include <unordered_map>

namespace lstr
{
class NodeLocalGlobalConverter
{
public:
    NodeLocalGlobalConverter(const MeshPartition* mesh_)
        : mesh{mesh_}, id_map{mesh_->getNodes().size() + mesh_->getGhostNodes().size()}
    {
        n_id_t local_ind = 0;
        for (auto gi : mesh->getNodes())
            id_map.insert(std::make_pair(gi, local_ind++));
        for (auto gi : mesh->getGhostNodes())
            id_map.insert(std::make_pair(gi, local_ind++));
    }

    [[nodiscard]] n_id_t globalToLocal(n_id_t global) const { return id_map.find(global)->second; }
    [[nodiscard]] n_id_t localToGlobal(n_id_t local) const
    {
        const auto n_owned_nodes = static_cast< n_id_t >(mesh->getNodes().size());
        return local < n_owned_nodes ? mesh->getNodes()[local] : mesh->getGhostNodes()[local - n_owned_nodes];
    }

    // The argument of the conversion functions must be the same as the one passed to the constructor
    void convertToLocal(MeshPartition& mesh_arg) const
    {
        mesh_arg.visit([this](auto& element) {
            for (auto& node : element.getNodes())
                node = globalToLocal(node);
        });
    }
    void convertToGlobal(MeshPartition& mesh_arg) const
    {
        mesh_arg.visit([this](auto& element) {
            for (auto& node : element.getNodes())
                node = localToGlobal(node);
        });
    }

private:
    const MeshPartition*                 mesh;
    std::unordered_map< n_id_t, n_id_t > id_map;
};
} // namespace lstr
#endif // L3STER_ASSEMBLY_NODELOCALGLOBALCONVERTER_HPP
