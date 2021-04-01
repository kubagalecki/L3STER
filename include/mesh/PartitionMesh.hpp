#ifndef L3STER_MESH_PARTITIONMESH_HPP
#define L3STER_MESH_PARTITIONMESH_HPP
#include "mesh/Mesh.hpp"
#include "util/Algorithm.hpp"

#include "metis.h"

#include <vector>

namespace lstr
{
inline void partitionMesh(Mesh& mesh, idx_t n_parts, const std::vector< d_id_t >& boundaries)
{
    if (mesh.getPartitions().size() != 1)
        throw std::logic_error{"Cannot partition a mesh which is either empty or has already been partitioned"};

    if (n_parts <= 1)
        return;

    MeshPartition& part            = mesh.getPartitions()[0];
    const auto     is_not_boundary = [&](const DomainView& dv) {
        return std::ranges::none_of(boundaries, [&](d_id_t b) { return b == dv.getID(); });
    };
    idx_t n_elements = 0, topology_size = 0, max_node = 0;
    part.cvisit(
        [&](const auto& el) {
            ++n_elements;
            topology_size += el.getNodes().size();
            const idx_t max_el_node = *std::ranges::max_element(el.getNodes());
            max_node                = std::max(max_node, max_el_node);
        },
        is_not_boundary);
    ++max_node;

    std::vector< idx_t > element_ids;
    std::vector< idx_t > e_ind, e_ptr, epart(n_elements), npart(max_node), node_comm_vol(max_node, 0),
        node_weight(max_node, 0);
    e_ind.reserve(topology_size);
    element_ids.reserve(n_elements);
    e_ptr.reserve(n_elements + 1);
    e_ptr.push_back(0);
    part.cvisit(
        [&](const auto& element) {
            constexpr auto element_size = std::tuple_size_v< std::decay_t< decltype(element.getNodes()) > >;
            std::ranges::for_each(element.getNodes(), [&](auto node) {
                e_ind.push_back(node);
                node_comm_vol[node] += element_size;
                node_weight[node] = 1;
            });
            e_ptr.push_back(e_ptr.back() + element_size);
            element_ids.push_back(element.getId());
        },
        is_not_boundary);

    idx_t      objval;
    const auto error = METIS_PartMeshNodal(&n_elements,
                                           &max_node,
                                           e_ptr.data(),
                                           e_ind.data(),
                                           node_weight.data(),
                                           node_comm_vol.data(),
                                           &n_parts,
                                           nullptr,
                                           nullptr,
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
    part.cvisit([&](const auto& element, const DomainView& dv) {
        if (is_not_boundary(dv))
            new_mesh[epart[index++]].pushElement(element, dv.getID());
    });

    {
        const auto sort_ind = sortingPermutation(element_ids.cbegin(), element_ids.cend());
        {
            decltype(element_ids) temp_el_ids(element_ids.size());
            copyPermuted(cbegin(element_ids), cend(element_ids), cbegin(sort_ind), begin(temp_el_ids));
            element_ids = std::move(temp_el_ids);
        }
        {
            decltype(epart) temp_epart(epart.size());
            copyPermuted(cbegin(epart), cend(epart), cbegin(sort_ind), begin(temp_epart));
            epart = std::move(temp_epart);
        }
    }
    const auto lookup_el_part = [&](size_t el_id) {
        return epart[std::distance(cbegin(element_ids),
                                   std::lower_bound(cbegin(element_ids), cend(element_ids), el_id))];
    };
    part.cvisit(
        [&](const auto& boundary_el, const DomainView& dv) {
            const auto domain_el = part.getElementBoundaryView(boundary_el, dv.getID()).first;
            const auto domain_part =
                lookup_el_part(std::visit([](const auto& el) { return el.get().getId(); }, *domain_el));
            new_mesh[domain_part].pushElement(boundary_el, dv.getID());
        },
        boundaries);

    mesh = Mesh{std::move(mesh.getNodes()), std::move(new_mesh)};
}
} // namespace lstr
#endif // L3STER_MESH_PARTITIONMESH_HPP
