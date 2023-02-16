#ifndef L3STER_MESH_PARTITIONMESH_HPP
#define L3STER_MESH_PARTITIONMESH_HPP

#include "l3ster/assembly/ProblemDefinition.hpp"
#include "l3ster/mesh/Mesh.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/MetisUtils.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

#include <vector>

// Note on naming: the uninformative names such as eptr, nparts, etc. are inherited from the METIS documentation
namespace lstr
{
namespace detail::part
{
template < ProblemDef_c auto problem_def >
auto computeNodeWeights(const MeshPartition& mesh, ConstexprValue< problem_def > probdef_ctwrapper)
    -> std::vector< idx_t >
{
    if constexpr (problem_def.size() == 0)
        return {};

    constexpr auto n_fields      = deduceNFields(problem_def);
    auto           node_dof_inds = DynamicBitset{n_fields * mesh.getAllNodes().size()};
    forConstexpr(
        [&]< auto dom_def >(ConstexprValue< dom_def >) {
            constexpr auto dom_id   = dom_def.first;
            constexpr auto dom_dofs = getTrueInds< dom_def.second >();
            mesh.visit(
                [&](const auto& element) {
                    for (auto node : element.getNodes())
                    {
                        auto node_dofs = node_dof_inds.getSubView(node * n_fields, (node + 1) * n_fields);
                        for (auto dof : dom_dofs)
                            node_dofs.set(dof);
                    }
                },
                dom_id);
        },
        probdef_ctwrapper);

    std::vector< idx_t > retval;
    retval.reserve(mesh.getAllNodes().size());
    std::ranges::transform(mesh.getAllNodes(), std::back_inserter(retval), [&](auto node) {
        const auto node_dofs = node_dof_inds.getSubView(node * n_fields, (node + 1) * n_fields);
        return node_dofs.count();
    });
    return retval;
}

inline auto getDomainPredicate(const std::vector< d_id_t >& boundaries)
{
    return [&](const DomainView& dv) {
        return std::ranges::none_of(boundaries, [&](d_id_t b) { return b == dv.getID(); });
    };
}

struct DomainData
{
    idx_t n_elements, topology_size, n_nodes;
};
template < std::invocable< const DomainView > F >
auto getDomainData(const MeshPartition& part, F&& domain_predicate) -> DomainData
{
    size_t n_elements = 0, topology_size = 0, max_node = 0;
    part.visit(
        [&](const auto& el) {
            ++n_elements;
            topology_size += el.getNodes().size();
            const auto max_el_node = *std::ranges::max_element(el.getNodes());
            max_node               = std::max(max_node, max_el_node);
        },
        std::forward< F >(domain_predicate));
    return {exactIntegerCast< idx_t >(n_elements),
            exactIntegerCast< idx_t >(topology_size),
            exactIntegerCast< idx_t >(++max_node)};
}

struct MetisInput
{
    std::vector< idx_t > e_ind, e_ptr;
};
auto prepMetisInput(const MeshPartition&                      part,
                    const DomainData&                         domain_data,
                    std::invocable< const DomainView > auto&& domain_predicate) -> MetisInput
{
    MetisInput retval;
    auto& [e_ind, e_ptr] = retval;
    e_ind.reserve(domain_data.topology_size);
    e_ptr.reserve(domain_data.n_elements + 1);
    e_ptr.push_back(0);
    part.visit(
        [&](const auto& element) {
            std::ranges::copy(element.getNodes(), std::back_inserter(e_ind));
            e_ptr.push_back(e_ptr.back() + element.getNodes().size());
        },
        std::forward< decltype(domain_predicate) >(domain_predicate));
    return retval;
}

inline auto getMetisOptionsForPartitioning()
{
    std::array< idx_t, METIS_NOPTIONS > opts{};
    METIS_SetDefaultOptions(opts.data());

    opts[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    opts[METIS_OPTION_CONTIG]  = 1;
    opts[METIS_OPTION_NCUTS]   = 2;
    opts[METIS_OPTION_NSEPS]   = 2;
    opts[METIS_OPTION_NITER]   = 20;

    return opts;
}

inline int invokeMetisPartitioner(idx_t                 n_elements,
                                  idx_t                 n_nodes,
                                  std::vector< idx_t >& epart,
                                  std::vector< idx_t >& npart,
                                  std::vector< idx_t >& e_ind,
                                  std::vector< idx_t >& e_ptr,
                                  std::vector< idx_t >& node_weights,
                                  idx_t                 n_parts,
                                  std::vector< real_t > part_weights)
{
    idx_t objval_discarded = 0;
    auto  metis_options    = getMetisOptionsForPartitioning();
    return METIS_PartMeshNodal(&n_elements,
                               &n_nodes,
                               e_ptr.data(),
                               e_ind.data(),
                               node_weights.empty() ? nullptr : node_weights.data(),
                               nullptr,
                               &n_parts,
                               part_weights.empty() ? nullptr : part_weights.data(),
                               metis_options.data(),
                               &objval_discarded,
                               epart.data(),
                               npart.data());
}

auto partitionDomains(const MeshPartition&                      part,
                      const DomainData&                         domain_data,
                      idx_t&                                    n_parts,
                      std::invocable< const DomainView > auto&& is_not_boundary,
                      std::vector< idx_t >                      node_weights,
                      std::vector< real_t >                     part_weights)
{
    auto [e_ind, e_ptr] = prepMetisInput(part, domain_data, std::forward< decltype(is_not_boundary) >(is_not_boundary));
    std::vector< idx_t > epart(domain_data.n_elements), npart(domain_data.n_nodes);
    const int            error_code = invokeMetisPartitioner(domain_data.n_elements,
                                                  domain_data.n_nodes,
                                                  epart,
                                                  npart,
                                                  e_ind,
                                                  e_ptr,
                                                  node_weights,
                                                  n_parts,
                                                  std::move(part_weights));
    detail::handleMetisErrorCode(error_code);
    return std::make_pair(std::move(epart), std::move(npart));
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
    element_ids         = getPermutedVector(sort_ind, element_ids);
    epart               = getPermutedVector(sort_ind, epart);
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

// Fix behavior where METIS sometimes ends up putting nodes in partitions where no elements contain them
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
        robin_hood::unordered_flat_set< idx_t > owned_nodes, ghost_nodes;
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
        constexpr auto vec_from_set = [](const robin_hood::unordered_flat_set< idx_t >& set) {
            std::vector< n_id_t > vec;
            vec.reserve(set.size());
            std::ranges::copy(set, std::back_inserter(vec));
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
} // namespace detail::part

template < RangeOfConvertibleTo_c< real_t > PartWgtRange = std::array< real_t, 0 >,
           detail::ProblemDef_c auto        problem_def  = detail::empty_problem_def_t{} >
Mesh partitionMesh(const Mesh&                   mesh,
                   idx_t                         n_parts,
                   const std::vector< d_id_t >&  boundaries,
                   PartWgtRange&&                part_weights   = {},
                   ConstexprValue< problem_def > probdef_ctwrpr = {})
{
    L3STER_PROFILE_FUNCTION;
    if (mesh.getPartitions().size() != 1)
        throw std::logic_error{"Cannot partition a mesh which is either empty or has already been partitioned"};

    if (n_parts <= 1)
        return mesh;

    std::vector< real_t > part_weights_converted;
    std::ranges::copy(std::forward< PartWgtRange >(part_weights), std::back_inserter(part_weights_converted));

    using namespace lstr::detail::part;
    const MeshPartition& part         = mesh.getPartitions().front();
    const auto           not_boundary = getDomainPredicate(boundaries);
    const auto           domain_data  = getDomainData(part, not_boundary);
    auto                 node_weights = computeNodeWeights(part, probdef_ctwrpr);
    auto [epart, npart]               = partitionDomains(
        part, domain_data, n_parts, not_boundary, std::move(node_weights), std::move(part_weights_converted));
    auto new_domain_maps = distributeDomainElements(part, n_parts, epart, not_boundary);
    assignBoundaryElements(part, epart, new_domain_maps, boundaries, domain_data.n_elements);
    auto node_vecs = assignNodes(n_parts, npart, new_domain_maps);
    return makeMeshFromPartitionComponents(std::move(new_domain_maps), std::move(node_vecs));
}
} // namespace lstr
#endif // L3STER_MESH_PARTITIONMESH_HPP
