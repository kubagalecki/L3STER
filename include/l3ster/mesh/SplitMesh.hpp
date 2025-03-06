#ifndef L3STER_MESH_SPLITMESH_HPP
#define L3STER_MESH_SPLITMESH_HPP

#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/CrsGraph.hpp"

namespace lstr::mesh
{
namespace detail
{
template < el_o_t... orders >
void assignBoundaryElements(const MeshPartition< orders... >&                                   mesh,
                            const robin_hood::unordered_flat_set< el_id_t >&                    true_el_ids,
                            std::array< typename MeshPartition< orders... >::domain_map_t, 2 >& domain_maps)
{
    auto& [true_els, false_els] = domain_maps;
    auto boundary_element_sizes = std::vector< size_t >{};
    for (auto boundary_id : mesh.getBoundaryIdsView())
    {
        const auto push_true_el_bnd_sz = [&]< ElementType ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el) {
            if (true_el_ids.contains(el->id))
                boundary_element_sizes.push_back(getSideNodeIndices< ET, EO >(el.getSide()).size());
        };
        mesh.visitBoundaries(push_true_el_bnd_sz, {boundary_id}, std::execution::seq);
    }
    auto boundary_el_nodes = util::CrsGraph< n_id_t >{boundary_element_sizes};
    auto true_boundary_el_nodes_set =
        robin_hood::unordered_flat_set< std::span< const n_id_t >,
                                        decltype([](std::span< const n_id_t > s) {
                                            return robin_hood::hash_bytes(s.data(), s.size_bytes());
                                        }),
                                        decltype([](std::span< const n_id_t > s1, std::span< const n_id_t > s2) {
                                            return std::ranges::equal(s1, s2);
                                        }) >{};
    for (size_t i = 0; auto boundary_id : mesh.getBoundaryIdsView())
    {
        const auto put_bnd_nodes_in_set = [&]< ElementType ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el) {
            if (true_el_ids.contains(el->id))
            {
                const auto dest = boundary_el_nodes(i++);
                std::ranges::copy(el.getSideNodesView(), dest.begin());
                std::ranges::sort(dest);
                true_boundary_el_nodes_set.insert(dest);
            }
        };
        mesh.visitBoundaries(put_bnd_nodes_in_set, {boundary_id}, std::execution::seq);
    }
    for (auto boundary_id : mesh.getBoundaryIdsView())
    {
        const auto push_elem = [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >& element) {
            auto nodes = element.nodes;
            std::ranges::sort(nodes);
            auto& dest = true_boundary_el_nodes_set.contains(std::span{nodes}) ? true_els : false_els;
            pushToDomain(dest[boundary_id], element);
        };
        mesh.visit(push_elem, {boundary_id}, std::execution::seq);
    }
}

template < typename ElementPredicate, el_o_t... orders >
auto makeDomainMaps(const MeshPartition< orders... >& mesh, ElementPredicate&& element_predicate)
{
    auto non_boundary_ids = std::vector< d_id_t >{};
    std::ranges::set_difference(mesh.getDomainIds(), mesh.getBoundaryIdsView(), std::back_inserter(non_boundary_ids));
    auto retval                 = std::array< typename MeshPartition< orders... >::domain_map_t, 2 >{};
    auto& [true_els, false_els] = retval;
    auto true_el_ids            = robin_hood::unordered_flat_set< el_id_t >{};
    for (auto domain_id : non_boundary_ids)
    {
        const auto push_elem = [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >& element) {
            if (std::invoke(element_predicate, element))
            {
                mesh::pushToDomain(true_els[domain_id], element);
                true_el_ids.insert(element.id);
            }
            else
                mesh::pushToDomain(false_els[domain_id], element);
        };
        mesh.visit(push_elem, domain_id);
    }
    assignBoundaryElements(mesh, true_el_ids, retval);
    return retval;
}

template < el_o_t... orders >
auto makeMesh(const MeshPartition< orders... >&                   parent_mesh,
              typename MeshPartition< orders... >::domain_map_t&& dom_map) -> MeshPartition< orders... >
{
    auto       owned_nodes        = std::set< n_id_t >{};
    const auto insert_owned_nodes = [&](const auto& element) {
        for (auto node : element.nodes | std::views::filter(
                                             [&](n_id_t node) { return parent_mesh.getNodeOwnership().isOwned(node); }))
            owned_nodes.insert(node);
    };
    for (const auto& domain : dom_map | std::views::values)
        domain.elements.visit(insert_owned_nodes, std::execution::seq);
    return {std::move(dom_map),
            owned_nodes.empty() ? 0uz : *owned_nodes.begin(),
            owned_nodes.size(),
            parent_mesh.getBoundaryIdsCopy()};
}
} // namespace detail

template < typename ElementPredicate, el_o_t... orders >
auto splitMeshPartition(const MeshPartition< orders... >& mesh, ElementPredicate&& element_predicate)
    -> std::array< MeshPartition< orders... >, 2 >
    requires ElementPredicate_c< ElementPredicate, orders... >
{
    auto [true_map, false_map] = detail::makeDomainMaps(mesh, std::forward< ElementPredicate >(element_predicate));
    return {detail::makeMesh(mesh, std::move(true_map)), detail::makeMesh(mesh, std::move(false_map))};
}
} // namespace lstr::mesh
#endif // L3STER_MESH_SPLITMESH_HPP
