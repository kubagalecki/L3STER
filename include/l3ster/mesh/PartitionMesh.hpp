#ifndef L3STER_MESH_PARTITIONMESH_HPP
#define L3STER_MESH_PARTITIONMESH_HPP

#include "l3ster/dofs/ProblemDefinition.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/MetisUtils.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

#include <vector>

// Note on naming: the uninformative names such as eptr, nparts, etc. are inherited from the METIS documentation
namespace lstr::mesh
{
namespace detail::part
{
auto convertPartWeights(RangeOfConvertibleTo_c< real_t > auto&& wgts) -> std::vector< real_t >
{
    std::vector< real_t > retval;
    retval.reserve(std::ranges::distance(wgts));
    std::ranges::copy(std::forward< decltype(wgts) >(wgts), std::back_inserter(retval));
    return retval;
}
inline auto convertPartWeights(std::vector< real_t > wgts) -> std::vector< real_t >
{
    return wgts;
}

template < el_o_t... orders, ProblemDef_c auto problem_def >
auto computeNodeWeights(const MeshPartition< orders... >& mesh, util::ConstexprValue< problem_def > probdef_ctwrapper)
    -> std::vector< idx_t >
{
    if constexpr (problem_def.size() == 0)
        return {};

    constexpr auto n_fields      = lstr::detail::deduceNFields(problem_def);
    auto           node_dof_inds = util::DynamicBitset{n_fields * mesh.getAllNodes().size()};
    util::forConstexpr(
        [&]< auto dom_def >(util::ConstexprValue< dom_def >) {
            constexpr auto dom_id   = dom_def.first;
            constexpr auto dom_dofs = util::getTrueInds< dom_def.second >();
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

template < el_o_t... orders >
auto getDomainIds(const MeshPartition< orders... >& mesh, const std::vector< d_id_t >& boundary_ids)
    -> std::vector< d_id_t >
{
    std::vector< d_id_t > retval;
    std::ranges::copy(mesh.getDomainIds() | std::views::filter([&](auto id) {
                          return std::ranges::find(boundary_ids, id) == end(boundary_ids);
                      }),
                      std::back_inserter(retval));
    std::ranges::sort(retval);
    return retval;
}

struct DomainData
{
    idx_t n_elements, topology_size;
};
template < el_o_t... orders >
auto getDomainData(const MeshPartition< orders... >& mesh, const std::vector< d_id_t >& domain_ids) -> DomainData
{
    size_t n_elements = 0, topology_size = 0;
    mesh.visit(
        [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >&) {
            ++n_elements;
            topology_size += ElementTraits< Element< ET, EO > >::boundary_node_inds.size();
        },
        domain_ids);
    return {util::exactIntegerCast< idx_t >(n_elements), util::exactIntegerCast< idx_t >(topology_size)};
}

struct MetisInput
{
    std::vector< idx_t > e_ind, e_ptr;
};
template < el_o_t... orders >
auto prepMetisInput(const MeshPartition< orders... >& part,
                    const DomainData&                 domain_data,
                    const std::vector< n_id_t >&      cond_map,
                    const std::vector< d_id_t >&      domain_ids) -> MetisInput
{
    MetisInput retval;
    auto& [e_ind, e_ptr] = retval;
    e_ind.reserve(domain_data.topology_size);
    e_ptr.reserve(domain_data.n_elements + 1);
    e_ptr.push_back(0);
    part.visit(
        [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >& element) {
            const auto condensed_view =
                getBoundaryNodes(element) | std::views::transform([&](auto n) { return cond_map[n]; });
            std::ranges::copy(condensed_view, std::back_inserter(e_ind));
            e_ptr.push_back(static_cast< idx_t >(e_ptr.back() + std::ranges::ssize(condensed_view)));
        },
        domain_ids);
    return retval;
}

template < el_o_t... orders >
auto makeNodeCondensationMaps(const MeshPartition< orders... >& mesh) -> std::array< std::vector< n_id_t >, 2 >
{
    std::vector< n_id_t > forward_map(mesh.getAllNodes().size()), reverse_map;
    mesh.visit(
        [&forward_map]< ElementType T, el_o_t O >(const Element< T, O >& element) {
            for (auto node : getBoundaryNodes(element))
                std::atomic_ref{forward_map[node]}.store(1, std::memory_order_relaxed);
        },
        std::execution::par);
    reverse_map.reserve(std::ranges::count(forward_map, 1));
    for (n_id_t node_full = 0, node_cond = 0; auto& forward_map_elem : forward_map)
    {
        if (forward_map_elem)
        {
            forward_map_elem = node_cond;
            reverse_map.push_back(node_full);
            ++node_cond;
        }
        ++node_full;
    }
    return {std::move(forward_map), std::move(reverse_map)};
}

inline auto condenseNodeWeights(std::vector< idx_t > node_weights, const std::vector< n_id_t >& reverse_map)
    -> std::vector< idx_t >
{
    if (not node_weights.empty())
        for (size_t node_cond = 0; auto node_uncond : reverse_map)
            node_weights[node_cond++] = node_weights[node_uncond];
    return node_weights;
}

inline auto makeMetisOptionsForPartitioning()
{
    std::array< idx_t, METIS_NOPTIONS > opts{};
    METIS_SetDefaultOptions(opts.data());
    opts[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    opts[METIS_OPTION_CONTIG]  = 1;
    opts[METIS_OPTION_NCUTS]   = 2;
    opts[METIS_OPTION_NSEPS]   = 2;
    opts[METIS_OPTION_NITER]   = 20;
    return opts;
}

struct MetisOutput
{
    std::vector< idx_t > epart, npart;
};
inline auto invokeMetisPartitioner(idx_t                 n_elements,
                                   idx_t                 n_nodes,
                                   MetisInput            metis_input,
                                   std::vector< idx_t >  node_weights,
                                   std::vector< real_t > part_weights,
                                   idx_t                 n_parts) -> MetisOutput
{
    auto        metis_options = makeMetisOptionsForPartitioning();
    idx_t       objval_discarded{};
    MetisOutput retval;
    retval.epart.resize(n_elements);
    retval.npart.resize(n_nodes);
    const auto error_code = METIS_PartMeshNodal(&n_elements,
                                                &n_nodes,
                                                metis_input.e_ptr.data(),
                                                metis_input.e_ind.data(),
                                                node_weights.empty() ? nullptr : node_weights.data(),
                                                node_weights.empty() ? nullptr : node_weights.data(),
                                                &n_parts,
                                                part_weights.empty() ? nullptr : part_weights.data(),
                                                metis_options.data(),
                                                &objval_discarded,
                                                retval.epart.data(),
                                                retval.npart.data());
    util::metis::handleMetisErrorCode(error_code);
    return retval;
}

template < el_o_t... orders >
auto uncondenseNodes(const std::vector< idx_t >&       epart,
                     const std::vector< idx_t >&       npart_cond,
                     const std::vector< n_id_t >&      reverse_map,
                     const MeshPartition< orders... >& mesh,
                     const std::vector< d_id_t >&      domain_ids) -> std::vector< idx_t >
{
    std::vector< idx_t > retval(mesh.getAllNodes().size());
    for (size_t i = 0; auto node_uncond : reverse_map)
        retval[node_uncond] = npart_cond[i++];
    size_t el_ind = 0;
    for (auto id : domain_ids)
        mesh.visit(
            [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >& element) {
                const auto el_part = epart[el_ind++];
                for (auto n : getInternalNodes(element))
                    retval[n] = el_part;
            },
            id);
    return retval;
}

template < el_o_t... orders >
auto partitionCondensedMesh(const MeshPartition< orders... >& mesh,
                            const std::vector< d_id_t >&      domain_ids,
                            DomainData                        domain_data,
                            idx_t                             n_parts,
                            std::vector< real_t >             part_weights,
                            std::vector< idx_t >              node_weights) -> MetisOutput
{
    const auto [forward_map, reverse_map] = makeNodeCondensationMaps(mesh);
    node_weights                          = condenseNodeWeights(std::move(node_weights), reverse_map);
    auto retval                           = invokeMetisPartitioner(domain_data.n_elements,
                                         util::exactIntegerCast< idx_t >(reverse_map.size()),
                                         prepMetisInput(mesh, domain_data, forward_map, domain_ids),
                                         std::move(node_weights),
                                         std::move(part_weights),
                                         n_parts);
    retval.npart                          = uncondenseNodes(retval.epart, retval.npart, reverse_map, mesh, domain_ids);
    return retval;
}

template < el_o_t... orders >
auto makeDomainMaps(const MeshPartition< orders... >& part,
                    idx_t                             n_parts,
                    const std::vector< idx_t >&       epart,
                    const std::vector< d_id_t >&      domain_ids)
{
    auto   retval = std::vector< typename MeshPartition< orders... >::domain_map_t >(n_parts);
    size_t index  = 0;
    for (auto id : domain_ids)
        part.visit([&](const auto& element) { retval[epart[index++]][id].push(element); }, id);
    return retval;
}

template < typename T >
std::vector< T > getPermutedVector(const std::vector< size_t >& perm, const std::vector< T >& input)
{
    std::vector< T > ret(input.size());
    util::copyPermuted(input.cbegin(), input.cend(), perm.cbegin(), ret.begin());
    return ret;
}

template < el_o_t... orders >
auto getElementIds(const MeshPartition< orders... >& part, size_t n_elements, const std::vector< d_id_t >& domain_ids)
    -> std::vector< el_id_t >
{
    std::vector< el_id_t > element_ids;
    element_ids.reserve(n_elements);
    part.visit([&](const auto& element) { element_ids.push_back(element.getId()); }, domain_ids);
    return element_ids;
}

inline void sortElementsById(std::vector< el_id_t >& element_ids, std::vector< idx_t >& epart)
{
    const auto sort_ind = util::sortingPermutation(element_ids.cbegin(), element_ids.cend());
    element_ids         = getPermutedVector(sort_ind, element_ids);
    epart               = getPermutedVector(sort_ind, epart);
}

template < el_o_t... orders >
void assignBoundaryElements(const MeshPartition< orders... >&                                 part,
                            std::vector< idx_t >&                                             epart,
                            std::vector< typename MeshPartition< orders... >::domain_map_t >& new_domain_maps,
                            const std::vector< d_id_t >&                                      domain_ids,
                            const std::vector< d_id_t >&                                      boundary_ids,
                            size_t                                                            n_elements)
{
    auto element_ids = getElementIds(part, n_elements, domain_ids);
    sortElementsById(element_ids, epart);
    const auto lookup_el_part = [&](size_t el_id) {
        return epart[std::distance(cbegin(element_ids),
                                   std::lower_bound(cbegin(element_ids), cend(element_ids), el_id))];
    };
    part.visit(
        [&](const auto& boundary_el, DomainView< orders... > dv) {
            const auto domain_el   = part.getElementBoundaryView(boundary_el, dv.getID()).first->first;
            const auto domain_part = lookup_el_part(std::visit([](const auto& el) { return el->getId(); }, domain_el));
            new_domain_maps[domain_part][dv.getID()].push(boundary_el);
        },
        boundary_ids);
}

// Fix behavior where METIS sometimes ends up putting nodes in partitions where no elements contain them
inline void reassignDisjointNodes(std::vector< std::pair< std::vector< n_id_t >, std::vector< n_id_t > > >& part_nodes,
                                  std::vector< idx_t >& disjoint_nodes)
{
    const auto claim_nodes = [&](std::pair< std::vector< n_id_t >, std::vector< n_id_t > >& nodes) {
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
    for (auto& nodes : part_nodes)
        claim_nodes(nodes);
    util::throwingAssert(disjoint_nodes.empty(), "At least one node in the mesh does not belong to any element");
}

template < el_o_t... orders >
auto assignNodes(idx_t                                                         n_parts,
                 const std::vector< idx_t >&                                   npart,
                 const std::vector< std::map< d_id_t, Domain< orders... > > >& domain_maps)
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
                            owned_nodes.insert(static_cast< idx_t >(node));
                        else
                            ghost_nodes.insert(static_cast< idx_t >(node));
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

template < el_o_t... orders >
auto makeMeshFromPartitionComponents(
    std::vector< std::map< d_id_t, Domain< orders... > > >&&                   dom_maps,
    std::vector< std::pair< std::vector< n_id_t >, std::vector< n_id_t > > >&& node_vecs)
    -> std::vector< MeshPartition< orders... > >
{
    std::vector< MeshPartition< orders... > > retval;
    retval.reserve(dom_maps.size());
    for (size_t i = 0; auto& [owned, ghost] : node_vecs)
        retval.emplace_back(std::move(dom_maps[i++]), std::move(owned), std::move(ghost));
    return retval;
}

template < el_o_t... orders >
auto partitionMeshImpl(const MeshPartition< orders... >& mesh,
                       idx_t                             n_parts,
                       const std::vector< d_id_t >&      boundary_ids,
                       std::vector< real_t >             part_weights,
                       std::vector< idx_t >              node_weights) -> std::vector< MeshPartition< orders... > >
{
    const auto domain_ids  = getDomainIds(mesh, boundary_ids);
    const auto domain_data = getDomainData(mesh, domain_ids);
    auto [epart, npart]    = partitionCondensedMesh(
        mesh, domain_ids, domain_data, n_parts, std::move(part_weights), std::move(node_weights));
    auto new_domain_maps = makeDomainMaps(mesh, n_parts, epart, domain_ids);
    assignBoundaryElements(mesh, epart, new_domain_maps, domain_ids, boundary_ids, domain_data.n_elements);
    auto node_vecs = assignNodes(n_parts, npart, new_domain_maps);
    return makeMeshFromPartitionComponents(std::move(new_domain_maps), std::move(node_vecs));
}
} // namespace detail::part

template < el_o_t... orders,
           RangeOfConvertibleTo_c< real_t > PartWgtRange = std::array< real_t, 0 >,
           ProblemDef_c auto                problem_def  = EmptyProblemDef{} >
auto partitionMesh(const MeshPartition< orders... >&   mesh,
                   idx_t                               n_parts,
                   const std::vector< d_id_t >&        boundary_ids,
                   PartWgtRange&&                      part_weights   = {},
                   util::ConstexprValue< problem_def > probdef_ctwrpr = {}) -> std::vector< MeshPartition< orders... > >
{
    L3STER_PROFILE_FUNCTION;
    util::throwingAssert(mesh.getGhostNodes().empty() and
                             mesh.getOwnedNodes().back() == mesh.getOwnedNodes().size() - 1,
                         "You cannot partition a mesh which has already been partitioned");

    if (n_parts <= 1)
        return {mesh};

    return detail::part::partitionMeshImpl(mesh,
                                           n_parts,
                                           boundary_ids,
                                           detail::part::convertPartWeights(std::forward< PartWgtRange >(part_weights)),
                                           detail::part::computeNodeWeights(mesh, probdef_ctwrpr));
}
} // namespace lstr::mesh
#endif // L3STER_MESH_PARTITIONMESH_HPP
