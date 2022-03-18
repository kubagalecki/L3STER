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
    NodeLocalGlobalConverter() = default;
    NodeLocalGlobalConverter(const MeshPartition& mesh) : id_map{mesh.getNodes().size() + mesh.getGhostNodes().size()}
    {
        n_id_t local_ind = 0;
        for (auto gi : mesh.getNodes())
            id_map.emplace(gi, local_ind++);
        for (auto gi : mesh.getGhostNodes())
            id_map.emplace(gi, local_ind++);
    }

    void convertToLocal(MeshPartition& mesh) const
    {
        mesh.visit([this](auto& element) {
            for (auto& node : element.getNodes())
                node = id_map.find(node)->second;
        });
    }
    void convertToGlobal(MeshPartition& mesh) const
    {
        const auto n_owned = static_cast< n_id_t >(mesh.getNodes().size());
        mesh.visit([&](auto& element) {
            for (auto& node : element.getNodes())
                node = node < n_owned ? mesh.getNodes()[node] : mesh.getGhostNodes()[node - n_owned];
        });
    }

private:
    std::unordered_map< n_id_t, n_id_t > id_map;
};
} // namespace lstr
#endif // L3STER_ASSEMBLY_NODELOCALGLOBALCONVERTER_HPP
