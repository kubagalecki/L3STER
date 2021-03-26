#ifndef L3STER_MESH_PARTITIONMESH_HPP
#define L3STER_MESH_PARTITIONMESH_HPP
#include "mesh/Mesh.hpp"

#include "metis.h"

#include <vector>

namespace lstr
{
// namespace detail
//{
// auto getElementSerializer()
//{}
//} // namespace detail

inline void partitionMesh(Mesh& mesh, idx_t n_parts, const std::vector< d_id_t >& boundaries)
{
    if (mesh.getPartitions().size() != 1)
        throw std::logic_error{"Cannot partition a mesh which is either empty or has already been partitioned"};

    if (n_parts <= 1)
        return;

    MeshPartition&       part       = mesh.getPartitions()[0];
    idx_t                n_elements = part.getNElements();
    idx_t                n_nodes    = mesh.getNodes().size();
    std::vector< idx_t > e_ind, e_ptr;
    e_ind.reserve(n_elements * 4 + boundaries.size()); // TODO: better initial node list size estimate
    e_ptr.reserve(n_elements + 1);
    e_ptr.push_back(0);
    part.cvisit([&](const auto& element) {
        std::ranges::for_each(element.getNodes(), [&](auto node) { e_ind.push_back(node); });
        e_ptr.push_back(e_ptr.back() + element.getNodes().size());
    });
    e_ind.shrink_to_fit();

    idx_t opts[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opts);
    std::vector< idx_t > epart(n_elements), npart(n_nodes);
    idx_t                objval;
    const auto           error = METIS_PartMeshNodal(&n_elements,
                                           &n_nodes,
                                           e_ptr.data(),
                                           e_ind.data(),
                                           nullptr,
                                           nullptr,
                                           &n_parts,
                                           nullptr,
                                           opts,
                                           &objval,
                                           epart.data(),
                                           npart.data());
    switch (error)
    {
    case METIS_OK:
        break;
    case METIS_ERROR_MEMORY:
        throw std::bad_alloc{};
    default:
        throw std::runtime_error{"Metis failed to partition the mesh"};
    }

    std::vector< MeshPartition > new_mesh(n_parts);
    size_t                       index = 0;
    part.cvisit(
        [&](const auto& element, const DomainView& dv) { new_mesh[epart[index++]].pushElement(element, dv.getID()); });

    mesh = Mesh{std::move(mesh.getNodes()), std::move(new_mesh)};
}
} // namespace lstr
#endif // L3STER_MESH_PARTITIONMESH_HPP
