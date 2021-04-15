#ifndef L3STER_MESH_PARTITIONMESH_HPP
#define L3STER_MESH_PARTITIONMESH_HPP
#include "mesh/Mesh.hpp"
#include "util/Algorithm.hpp"
#include "util/MetisUtils.hpp"

#include <vector>

namespace lstr
{
namespace detail
{
inline auto getDomainPredicate(const std::vector< d_id_t >& boundaries)
{
    return [&](const DomainView& dv) {
        return std::ranges::none_of(boundaries, [&](d_id_t b) { return b == dv.getID(); });
    };
}

template < std::invocable< const DomainView > F >
std::array< idx_t, 3 > getDomainData(const MeshPartition& part, F&& domain_predicate)
{
    idx_t n_elements = 0, topology_size = 0, max_node = 0;
    part.cvisit(
        [&](const auto& el) {
            ++n_elements;
            topology_size += el.getNodes().size();
            const idx_t max_el_node = *std::ranges::max_element(el.getNodes());
            max_node                = std::max(max_node, max_el_node);
        },
        std::forward< F >(domain_predicate));
    return {n_elements, topology_size, max_node + 1};
}

template < std::invocable< const DomainView > F >
auto prepMetisInput(const MeshPartition& part, const std::array< idx_t, 3 >& domain_data, F&& domain_predicate)
{
    const auto& [n_elements, topology_size, max_node] = domain_data;
    std::vector< idx_t > e_ind, e_ptr, node_comm_vol(max_node, 0), node_weight(max_node, 0);
    e_ind.reserve(topology_size);
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
        },
        std::forward< F >(domain_predicate));
    return std::make_tuple(std::move(e_ind), std::move(e_ptr), std::move(node_comm_vol), std::move(node_weight));
}

inline int invokeMetis(idx_t&                n_elements,
                       idx_t&                max_node,
                       std::vector< idx_t >& epart,
                       std::vector< idx_t >& npart,
                       std::vector< idx_t >& e_ind,
                       std::vector< idx_t >& e_ptr,
                       std::vector< idx_t >& node_comm_vol,
                       std::vector< idx_t >& node_weight,
                       idx_t&                n_parts)
{
    idx_t objval;
    return METIS_PartMeshNodal(&n_elements,
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
}

template < std::invocable< const DomainView > F >
auto partitionDomains(const MeshPartition&          part,
                      const std::array< idx_t, 3 >& domain_data,
                      idx_t&                        n_parts,
                      F&&                           is_not_boundary)
{
    idx_t n_elements = domain_data[0], max_node = domain_data[2];
    auto  metis_input = detail::prepMetisInput(part, domain_data, std::forward< F >(is_not_boundary));
    auto& [e_ind, e_ptr, node_comm_vol, node_weight] = metis_input;
    std::vector< idx_t > epart(n_elements), npart(max_node);
    const int            error_code =
        detail::invokeMetis(n_elements, max_node, epart, npart, e_ind, e_ptr, node_comm_vol, node_weight, n_parts);
    detail::handleMetisErrorCode(error_code);
    return std::make_tuple(std::move(epart), std::move(npart));
}

auto distributeDomainElements(const MeshPartition&                      part,
                              idx_t                                     n_parts,
                              const std::vector< idx_t >&               epart,
                              std::invocable< const DomainView > auto&& domain_predicate)
{
    std::vector< MeshPartition::domain_map_t > new_domain_maps(n_parts);
    part.cvisit([&, index = 0u](const auto& element, const DomainView& dv) mutable {
        if (domain_predicate(dv))
            new_domain_maps[epart[index++]][dv.getID()].push(element);
    });
    return new_domain_maps;
}

template < typename T >
std::vector< T > getPermutedVector(const std::vector< size_t >& perm, const std::vector< T >& input)
{
    std::vector< T > ret(input.size());
    copyPermuted(input.cbegin(), input.cend(), perm.cbegin(), ret.begin());
    return ret;
}

template < std::invocable< const DomainView > F >
std::vector< el_id_t > getElementIds(const MeshPartition& part, size_t n_elements, F&& domain_predicate)
{
    std::vector< el_id_t > element_ids;
    element_ids.reserve(n_elements);
    part.cvisit([&](const auto& element) { element_ids.push_back(element.getId()); },
                std::forward< F >(domain_predicate));
    return element_ids;
}

inline void sortElementsById(std::vector< el_id_t >& element_ids, std::vector< idx_t >& epart)
{
    const auto sort_ind = sortingPermutation(element_ids.cbegin(), element_ids.cend());
    element_ids         = detail::getPermutedVector(sort_ind, element_ids);
    epart               = detail::getPermutedVector(sort_ind, epart);
}

inline void assignBoundaryElements(const MeshPartition&                        part,
                                   std::vector< idx_t >&                       epart,
                                   std::vector< MeshPartition::domain_map_t >& new_domain_maps,
                                   const std::vector< d_id_t >&                boundaries,
                                   size_t                                      n_elements)
{
    const auto is_not_boundary = getDomainPredicate(boundaries);
    auto       element_ids     = getElementIds(part, n_elements, is_not_boundary);
    sortElementsById(element_ids, epart);
    const auto lookup_el_part = [&](size_t el_id) {
        return epart[std::distance(cbegin(element_ids),
                                   std::lower_bound(cbegin(element_ids), cend(element_ids), el_id))];
    };
    part.cvisit(
        [&](const auto& boundary_el, const DomainView& dv) {
            const auto domain_el   = part.getElementBoundaryView(boundary_el, dv.getID()).first;
            const auto domain_part = lookup_el_part(std::visit([](const auto& el) { return el->getId(); }, *domain_el));
            new_domain_maps[domain_part][dv.getID()].push(boundary_el);
        },
        boundaries);
}

inline std::vector< MeshPartition > assignNodes(idx_t                                       n_parts,
                                                const std::vector< idx_t >&                 npart,
                                                std::vector< MeshPartition::domain_map_t >& new_domain_maps)
{
    std::vector< MeshPartition > new_partitions;
    new_partitions.reserve(n_parts);
    idx_t               part_ind = 0;
    std::vector< char > node_types(npart.size());
    for (auto&& dm : new_domain_maps)
    {
        constexpr char none = 0, normal = 1, ghost = 2;
        size_t         n_normal = 0, n_ghosts = 0;
        std::ranges::fill(node_types, none);
        std::ranges::for_each(dm, [&](const auto& dom_p) {
            dom_p.second.cvisit([&](const auto& element) {
                std::ranges::for_each(element.getNodes(), [&](n_id_t node) {
                    if (npart[node] == part_ind)
                    {
                        node_types[node] = normal;
                        ++n_normal;
                    }
                    else
                    {
                        node_types[node] = ghost;
                        ++n_ghosts;
                    }
                });
            });
        });
        std::vector< n_id_t > nodes, ghost_nodes;
        nodes.reserve(n_normal);
        ghost_nodes.reserve(n_ghosts);
        std::ranges::for_each(node_types, [&, index = 0](char type) mutable {
            switch (type)
            {
            case normal:
                nodes.push_back(index);
                break;
            case ghost:
                ghost_nodes.push_back(index);
                break;
            default:
                break;
            }
            ++index;
        });

        new_partitions.emplace_back(std::move(dm), std::move(nodes), std::move(ghost_nodes));
        ++part_ind;
    }
    return new_partitions;
}
} // namespace detail

inline void partitionMesh(Mesh& mesh, idx_t n_parts, const std::vector< d_id_t >& boundaries)
{
    if (mesh.getPartitions().size() != 1)
        throw std::logic_error{"Cannot partition a mesh which is either empty or has already been partitioned"};

    if (n_parts <= 1)
        return;

    const MeshPartition& part            = mesh.getPartitions()[0];
    const auto           is_not_boundary = detail::getDomainPredicate(boundaries);
    const auto           domain_data     = detail::getDomainData(part, is_not_boundary);

    auto metis_result    = detail::partitionDomains(part, domain_data, n_parts, is_not_boundary);
    auto& [epart, npart] = metis_result;

    auto new_domain_maps = detail::distributeDomainElements(part, n_parts, epart, is_not_boundary);
    detail::assignBoundaryElements(part, epart, new_domain_maps, boundaries, domain_data[0]);
    auto new_partitions = detail::assignNodes(n_parts, npart, new_domain_maps);

    mesh = Mesh{std::move(mesh.getVertices()), std::move(new_partitions)};
}
} // namespace lstr
#endif // L3STER_MESH_PARTITIONMESH_HPP
