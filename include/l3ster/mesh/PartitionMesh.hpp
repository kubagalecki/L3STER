#ifndef L3STER_MESH_PARTITIONMESH_HPP
#define L3STER_MESH_PARTITIONMESH_HPP

#include "l3ster/mesh/Mesh.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/MetisUtils.hpp"

#include <unordered_set>
#include <vector>

namespace lstr
{
// Note on naming: the uninformative names such as eptr, nparts, etc. are inherited from the METIS documentation
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
    part.visit(
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
    part.visit(
        [&](const auto& element) {
            constexpr auto element_size = std::tuple_size_v< std::decay_t< decltype(element.getNodes()) > >;
            for (auto node : element.getNodes())
            {
                e_ind.push_back(node);
                node_comm_vol[node] += element_size;
                node_weight[node] = 1;
            }
            e_ptr.push_back(e_ptr.back() + element_size);
        },
        std::forward< F >(domain_predicate));
    return std::make_tuple(std::move(e_ind), std::move(e_ptr), std::move(node_comm_vol), std::move(node_weight));
}

inline auto getMetisOptionsForPartitioning()
{
    std::array< idx_t, METIS_NOPTIONS > opts{};
    METIS_SetDefaultOptions(opts.data());

    opts[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    opts[METIS_OPTION_CONTIG]  = 1;
    opts[METIS_OPTION_NCUTS]   = 3;
    opts[METIS_OPTION_NSEPS]   = 3;
    opts[METIS_OPTION_NITER]   = 20;

    return opts;
}

inline int invokeMetisPartitioner(idx_t&                n_elements,
                                  idx_t&                max_node,
                                  std::vector< idx_t >& epart,
                                  std::vector< idx_t >& npart,
                                  std::vector< idx_t >& e_ind,
                                  std::vector< idx_t >& e_ptr,
                                  std::vector< idx_t >& node_comm_vol,
                                  std::vector< idx_t >& node_weight,
                                  idx_t&                n_parts,
                                  std::vector< real_t > part_weights)
{
    idx_t objval_discarded = 0;
    auto  metis_options    = getMetisOptionsForPartitioning();
    return METIS_PartMeshNodal(&n_elements,
                               &max_node,
                               e_ptr.data(),
                               e_ind.data(),
                               node_weight.data(),
                               node_comm_vol.data(),
                               &n_parts,
                               part_weights.empty() ? nullptr : part_weights.data(),
                               metis_options.data(),
                               &objval_discarded,
                               epart.data(),
                               npart.data());
}

template < std::invocable< const DomainView > F >
auto partitionDomains(const MeshPartition&          part,
                      const std::array< idx_t, 3 >& domain_data,
                      idx_t&                        n_parts,
                      F&&                           is_not_boundary,
                      std::vector< real_t >         part_weights)
{
    idx_t n_elements = domain_data[0], max_node = domain_data[2];
    auto  metis_input = detail::prepMetisInput(part, domain_data, std::forward< F >(is_not_boundary));
    auto& [e_ind, e_ptr, node_comm_vol, node_weight] = metis_input;
    std::vector< idx_t > epart(n_elements), npart(max_node);

    const int error_code = detail::invokeMetisPartitioner(
        n_elements, max_node, epart, npart, e_ind, e_ptr, node_comm_vol, node_weight, n_parts, std::move(part_weights));
    detail::handleMetisErrorCode(error_code);
    return std::make_tuple(std::move(epart), std::move(npart));
}

auto distributeDomainElements(const MeshPartition&                      part,
                              idx_t                                     n_parts,
                              const std::vector< idx_t >&               epart,
                              std::invocable< const DomainView > auto&& domain_predicate)
{
    std::vector< MeshPartition::domain_map_t > new_domain_maps(n_parts);
    part.visit([&, index = 0u](const auto& element, const DomainView& dv) mutable {
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
    part.visit([&](const auto& element) { element_ids.push_back(element.getId()); },
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
    part.visit(
        [&](const auto& boundary_el, const DomainView& dv) {
            const auto domain_el   = part.getElementBoundaryView(boundary_el, dv.getID()).first->first;
            const auto domain_part = lookup_el_part(std::visit([](const auto& el) { return el->getId(); }, domain_el));
            new_domain_maps[domain_part][dv.getID()].push(boundary_el);
        },
        boundaries);
}

inline void reassignDisjointNodes(std::vector< std::pair< std::vector< n_id_t >, std::vector< n_id_t > > >& part_nodes,
                                  std::vector< idx_t >& disjoint_nodes)
{
    const auto claim_nodes = [&](idx_t part, std::pair< std::vector< n_id_t >, std::vector< n_id_t > >& nodes) {
        std::vector< n_id_t > claimed;
        auto& [owned_nodes, ghost_nodes] = nodes;
        const auto try_claim             = [&](idx_t node) {
            const auto ghost_iter = std::ranges::lower_bound(ghost_nodes, node);
            if (ghost_iter == ghost_nodes.end() or *ghost_iter != static_cast< n_id_t >(node))
                return false;
            claimed.emplace_back(node);
            ghost_nodes.erase(ghost_iter);
            return true;
        };
        std::erase_if(disjoint_nodes, try_claim);
        std::ranges::sort(claimed);
        const auto old_n_owned = owned_nodes.size();
        owned_nodes.resize(old_n_owned + claimed.size());
        const auto insert_pos = std::next(begin(owned_nodes), static_cast< ptrdiff_t >(old_n_owned));
        std::ranges::copy(claimed, insert_pos);
        std::ranges::inplace_merge(owned_nodes, insert_pos);
    };
    for (idx_t part = 0; auto& nodes : part_nodes)
        claim_nodes(part++, nodes);
    if (not disjoint_nodes.empty())
        throw std::logic_error{"At least one node in the mesh does not belong to any element"};
}

inline auto assignNodes(idx_t                                             n_parts,
                        const std::vector< idx_t >&                       npart,
                        const std::vector< MeshPartition::domain_map_t >& domain_maps)
{
    std::vector< std::pair< std::vector< n_id_t >, std::vector< n_id_t > > > new_node_vecs;
    std::vector< idx_t >                                                     disjoint_nodes;
    new_node_vecs.reserve(n_parts);
    for (idx_t part_ind = 0; const auto& dom_map : domain_maps)
    {
        std::unordered_set< idx_t > owned_nodes, ghost_nodes;
        for (const auto& domain : dom_map | std::views::values)
        {
            domain.visit(
                [&](const auto& element) {
                    for (auto node : element.getNodes())
                        if (npart[node] == part_ind)
                            owned_nodes.insert(node);
                        else
                            ghost_nodes.insert(node);
                },
                std::execution::seq);
        }
        for (idx_t n = 0; auto p : npart)
        {
            if (p == part_ind and not owned_nodes.contains(n))
                disjoint_nodes.push_back(n);
            ++n;
        }
        constexpr auto vec_from_set = [](const std::unordered_set< idx_t >& set) {
            std::vector< n_id_t > vec(set.size());
            std::ranges::copy(set, begin(vec));
            std::ranges::sort(vec);
            return vec;
        };
        new_node_vecs.emplace_back(vec_from_set(owned_nodes), vec_from_set(ghost_nodes));
        ++part_ind;
    }
    reassignDisjointNodes(new_node_vecs, disjoint_nodes);
    return new_node_vecs;
}

inline Mesh
makeMeshFromPartitionComponents(std::vector< MeshPartition::domain_map_t >&&                               dom_maps,
                                std::vector< std::pair< std::vector< n_id_t >, std::vector< n_id_t > > >&& node_vecs)
{
    std::vector< MeshPartition > new_parts;
    new_parts.reserve(dom_maps.size());
    for (size_t i = 0; auto& [owned, ghost] : node_vecs) // can't use std::transform since we modify the source range
        new_parts.emplace_back(std::move(dom_maps[i++]), std::move(owned), std::move(ghost));
    return Mesh{std::move(new_parts)};
}
} // namespace detail

[[nodiscard]] inline Mesh partitionMesh(const Mesh&                  mesh,
                                        idx_t                        n_parts,
                                        const std::vector< d_id_t >& boundaries,
                                        std::vector< real_t >        part_weights = {})
{
    if (mesh.getPartitions().size() != 1)
        throw std::logic_error{"Cannot partition a mesh which is either empty or has already been partitioned"};

    if (n_parts <= 1)
        return mesh;

    const MeshPartition& part         = mesh.getPartitions()[0];
    const auto           not_boundary = detail::getDomainPredicate(boundaries);
    const auto           domain_data  = detail::getDomainData(part, not_boundary);
    auto [epart, npart]  = detail::partitionDomains(part, domain_data, n_parts, not_boundary, std::move(part_weights));
    auto new_domain_maps = detail::distributeDomainElements(part, n_parts, epart, not_boundary);
    detail::assignBoundaryElements(part, epart, new_domain_maps, boundaries, domain_data[0]);
    auto node_vecs = detail::assignNodes(n_parts, npart, new_domain_maps);
    return detail::makeMeshFromPartitionComponents(std::move(new_domain_maps), std::move(node_vecs));
}
} // namespace lstr
#endif // L3STER_MESH_PARTITIONMESH_HPP
