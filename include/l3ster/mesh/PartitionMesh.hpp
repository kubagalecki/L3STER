#ifndef L3STER_MESH_PARTITIONMESH_HPP
#define L3STER_MESH_PARTITIONMESH_HPP

#include "ElementTraits.hpp"
#include "l3ster/dofs/ProblemDefinition.hpp"
#include "l3ster/mesh/MeshUtils.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/MetisUtils.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

#include <vector>

// Note on naming: the uninformative names such as eptr, nparts, etc. are inherited from the METIS documentation
namespace lstr::mesh
{
struct PartitioningOpts
{
    bool renumber = true;
};

namespace detail
{
inline auto convertPartWeights(util::ArrayOwner< real_t > wgts) -> util::ArrayOwner< real_t >
{
    return wgts;
}

template < el_o_t... orders, ProblemDef problem_def >
auto computeNodeWeights(const MeshPartition< orders... >& mesh, util::ConstexprValue< problem_def > probdef_ctwrpr)
    -> util::ArrayOwner< idx_t >
{
    if constexpr (problem_def.n_domains == 0)
        return {};

    constexpr auto n_fields      = problem_def.n_fields;
    auto           node_dof_inds = util::DynamicBitset{problem_def.n_fields * mesh.getAllNodes().size()};
    util::forConstexpr(
        [&]< DomainDef< n_fields > dom_def >(util::ConstexprValue< dom_def >) {
            constexpr auto dom_dofs = util::getTrueInds< dom_def.active_fields >();
            mesh.visit(
                [&](const auto& element) {
                    for (auto node : element.getNodes())
                    {
                        const auto begin_ind      = node * problem_def.n_fields;
                        const auto end_ind        = begin_ind + problem_def.n_fields;
                        auto       node_dofs_view = node_dof_inds.getSubView(begin_ind, end_ind);
                        for (auto dof : dom_dofs)
                            node_dofs_view.set(dof);
                    }
                },
                dom_def.domain);
        },
        probdef_ctwrpr);

    auto retval = util::ArrayOwner< idx_t >(mesh.getAllNodes().size());
    std::ranges::transform(mesh.getAllNodes(), retval.begin(), [&](auto node) {
        const auto node_dofs = node_dof_inds.getSubView(node * problem_def.n_fields, (node + 1) * problem_def.n_fields);
        return node_dofs.count();
    });
    return retval;
}

template < el_o_t... orders >
auto getDomainIds(const MeshPartition< orders... >& mesh, const util::ArrayOwner< d_id_t >& boundary_ids)
    -> util::ArrayOwner< d_id_t >
{
    return mesh.getDomainIds() |
           std::views::filter([&](auto id) { return std::ranges::find(boundary_ids, id) == boundary_ids.end(); });
}

struct DomainData
{
    idx_t n_elements, topology_size;
};
template < el_o_t... orders >
auto getDomainData(const MeshPartition< orders... >& mesh, const util::ArrayOwner< d_id_t >& domain_ids) -> DomainData
{
    size_t     n_elements = 0, topology_size = 0;
    const auto count_el_data = [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >&) {
        ++n_elements;
        topology_size += ElementTraits< Element< ET, EO > >::boundary_node_inds.size();
    };
    mesh.visit(count_el_data, domain_ids);
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
                    const util::ArrayOwner< d_id_t >& domain_ids) -> MetisInput
{
    auto retval          = MetisInput{};
    auto& [e_ind, e_ptr] = retval;
    e_ind.reserve(domain_data.topology_size);
    e_ptr.reserve(domain_data.n_elements + 1);
    e_ptr.push_back(0);
    const auto condense = [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >& element) {
        const auto condensed_view =
            getBoundaryNodes(element) | std::views::transform([&](auto n) { return cond_map[n]; });
        std::ranges::copy(condensed_view, std::back_inserter(e_ind));
        e_ptr.push_back(static_cast< idx_t >(e_ptr.back() + std::ranges::ssize(condensed_view)));
    };
    part.visit(condense, domain_ids);
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

inline auto condenseNodeWeights(util::ArrayOwner< idx_t > node_weights, const util::ArrayOwner< n_id_t >& reverse_map)
    -> util::ArrayOwner< idx_t >
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
    util::ArrayOwner< idx_t > epart, npart;
};
inline auto invokeMetisPartitioner(idx_t                      n_els,
                                   idx_t                      n_nodes,
                                   MetisInput                 metis_input,
                                   util::ArrayOwner< idx_t >  node_weights,
                                   util::ArrayOwner< real_t > part_weights,
                                   idx_t                      n_parts) -> MetisOutput
{
    auto  metis_options = makeMetisOptionsForPartitioning();
    idx_t objval_discarded{};
    auto  retval = MetisOutput{.epart = util::ArrayOwner< idx_t >(n_els), .npart = util::ArrayOwner< idx_t >(n_nodes)};

    const auto error_code = METIS_PartMeshNodal(&n_els,
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
auto uncondenseNodes(const util::ArrayOwner< idx_t >&  epart,
                     const util::ArrayOwner< idx_t >&  npart_cond,
                     const std::vector< n_id_t >&      reverse_map,
                     const MeshPartition< orders... >& mesh,
                     const util::ArrayOwner< d_id_t >& domain_ids) -> util::ArrayOwner< idx_t >
{
    util::ArrayOwner< idx_t > retval(mesh.getAllNodes().size());
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
                            const util::ArrayOwner< d_id_t >& domain_ids,
                            DomainData                        domain_data,
                            idx_t                             n_parts,
                            util::ArrayOwner< real_t >        part_weights,
                            util::ArrayOwner< idx_t >         node_weights) -> MetisOutput
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
                    const util::ArrayOwner< idx_t >&  epart,
                    const util::ArrayOwner< d_id_t >& domain_ids)
{
    auto retval = util::ArrayOwner< typename MeshPartition< orders... >::domain_map_t >(n_parts);
    for (size_t index = 0; auto id : domain_ids)
    {
        const auto push_to_domain_map = [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >& element) {
            const auto target_partition = epart.at(index++);
            auto&      domain           = retval.at(target_partition)[id];
            pushToDomain(domain, element);
        };
        part.visit(push_to_domain_map, id);
    }
    return retval;
}

template < typename T >
util::ArrayOwner< T > getPermutedVector(const util::ArrayOwner< size_t >& perm, const util::ArrayOwner< T >& input)
{
    auto ret = util::ArrayOwner< T >(input.size());
    util::copyPermuted(input.begin(), input.end(), perm.begin(), ret.begin());
    return ret;
}

template < el_o_t... orders >
auto getElementIds(const MeshPartition< orders... >& part,
                   size_t                            n_elements,
                   const util::ArrayOwner< d_id_t >& domain_ids) -> util::ArrayOwner< el_id_t >
{
    auto   element_ids = util::ArrayOwner< el_id_t >(n_elements);
    size_t i           = 0;
    part.visit([&](const auto& element) { element_ids[i++] = (element.getId()); }, domain_ids);
    return element_ids;
}

inline void sortElementsById(util::ArrayOwner< el_id_t >& element_ids, util::ArrayOwner< idx_t >& epart)
{
    const auto sort_ind = util::sortingPermutation(element_ids.begin(), element_ids.end());
    element_ids         = getPermutedVector(sort_ind, element_ids);
    epart               = getPermutedVector(sort_ind, epart);
}

template < el_o_t... orders >
void assignBoundaryElements(const MeshPartition< orders... >&                                      part,
                            util::ArrayOwner< idx_t >&                                             epart,
                            util::ArrayOwner< typename MeshPartition< orders... >::domain_map_t >& new_domain_maps,
                            const util::ArrayOwner< d_id_t >&                                      domain_ids,
                            const util::ArrayOwner< d_id_t >&                                      boundary_ids,
                            size_t                                                                 n_elements)
{
    auto element_ids = getElementIds(part, n_elements, domain_ids);
    sortElementsById(element_ids, epart);
    const auto lookup_el_part = [&](size_t el_id) {
        const auto el_it  = std::lower_bound(element_ids.begin(), element_ids.end(), el_id);
        const auto el_pos = std::distance(element_ids.begin(), el_it);
        return epart.at(el_pos);
    };

    const auto         dim_dom_map = detail::makeDimToDomainMap(part);
    std::atomic_size_t n_boundary_elements{0}, n_assigned{0};
    auto               insertion_mutex = std::mutex{};
    const auto         assign_boundary = [&](d_id_t boundary_id) {
        const auto& boundary = part.getDomain(boundary_id);
        n_boundary_elements.fetch_add(boundary.elements.size(), std::memory_order_relaxed);
        const auto assign_from_domain = [&](d_id_t domain_id) {
            const auto assign_boundary_els = [&]< ElementType ET, el_o_t EO >(
                                                 const Element< ET, EO >& boundary_element) {
                const auto dom_el_opt = findDomainElement(part, boundary_element, std::views::single(domain_id));
                if (dom_el_opt)
                {
                    n_assigned.fetch_add(1, std::memory_order_relaxed);
                    const auto dom_el_id = std::visit([](const auto& el) { return el->getId(); }, dom_el_opt->first);
                    const auto domain_element_partition = lookup_el_part(dom_el_id);

                    const auto lock   = std::lock_guard{insertion_mutex};
                    auto&      domain = new_domain_maps[domain_element_partition][boundary_id];
                    pushToDomain(domain, boundary_element);
                }
            };
            part.visit(assign_boundary_els, std::views::single(boundary_id));
        };
        const auto& potential_dom_ids = dim_dom_map.at(boundary.dim + 1);
        util::tbb::parallelFor(potential_dom_ids, assign_from_domain);
    };
    util::tbb::parallelFor(boundary_ids, assign_boundary);
    util::throwingAssert(
        n_assigned.load() == n_boundary_elements.load(),
        "The mesh partitioner could not assign all edge/face elements to the partitions of their corresponding "
        "area/volume elements. Make sure that you have correctly specified the boundary IDs.");
}

// Fix behavior where METIS sometimes ends up putting nodes in partitions where no elements contain them
inline void reassignDisjointNodes(util::ArrayOwner< std::array< std::vector< n_id_t >, 2 > >& part_nodes,
                                  std::vector< idx_t >&                                       disjoint_nodes)
{
    const auto claim_nodes = [&](std::array< std::vector< n_id_t >, 2 >& nodes) {
        auto claimed                     = std::vector< n_id_t >{};
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
auto assignNodes(idx_t                                                              n_parts,
                 const util::ArrayOwner< idx_t >&                                   npart,
                 const util::ArrayOwner< std::map< d_id_t, Domain< orders... > > >& domain_maps)
    -> util::ArrayOwner< std::array< std::vector< n_id_t >, 2 > >
{
    auto new_node_vecs  = util::ArrayOwner< std::array< std::vector< n_id_t >, 2 > >(n_parts);
    auto disjoint_nodes = std::vector< idx_t >{};
    for (idx_t part_ind = 0; const auto& dom_map : domain_maps)
    {
        robin_hood::unordered_flat_set< idx_t > owned_nodes, ghost_nodes;
        for (const auto& domain : dom_map | std::views::values)
        {
            domain.elements.visit(
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
            auto vec = std::vector< n_id_t >{};
            vec.reserve(set.size());
            std::ranges::copy(set, std::back_inserter(vec));
            std::ranges::sort(vec);
            return vec;
        };
        new_node_vecs[part_ind].front() = vec_from_set(owned_nodes);
        new_node_vecs[part_ind].back()  = vec_from_set(ghost_nodes);
        ++part_ind;
    }
    reassignDisjointNodes(new_node_vecs, disjoint_nodes);
    return new_node_vecs;
}

template < el_o_t... orders >
auto makeMeshFromPartitionComponents(util::ArrayOwner< std::map< d_id_t, Domain< orders... > > >&& dom_maps,
                                     util::ArrayOwner< std::array< std::vector< n_id_t >, 2 > >&&  node_vecs,
                                     const util::ArrayOwner< d_id_t >&                             bnd_ids)
    -> util::ArrayOwner< MeshPartition< orders... > >
{
    auto retval = util::ArrayOwner< MeshPartition< orders... > >(dom_maps.size());
    for (size_t i = 0; auto& [owned, ghost] : node_vecs)
    {
        auto part = MeshPartition< orders... >{std::move(dom_maps[i]), std::move(owned), std::move(ghost), bnd_ids};
        retval[i] = std::move(part);
        ++i;
    }
    return retval;
}

template < el_o_t... orders >
void renumberNodes(util::ArrayOwner< std::map< d_id_t, Domain< orders... > > >& dom_maps,
                   util::ArrayOwner< std::array< std::vector< n_id_t >, 2 > >&  node_vecs)
{
    auto node_id_map = robin_hood::unordered_flat_map< n_id_t, n_id_t >{};
    for (n_id_t new_id = 0; auto& [owned, _] : node_vecs)
        for (auto& old_id : owned)
        {
            node_id_map.emplace(old_id, new_id);
            old_id = new_id++;
        }
    const auto update_id = [&node_id_map](n_id_t& id) {
        id = node_id_map.at(id);
    };
    for (auto& [_, ghost] : node_vecs)
    {
        std::ranges::for_each(ghost, update_id);
        std::ranges::sort(ghost);
    }
    const auto update_element = [&]< ElementType ET, el_o_t EO >(Element< ET, EO >& element) {
        std::ranges::for_each(element.getNodes(), update_id);
    };
    for (auto& map : dom_maps)
        for (auto& map_entry : map)
        {
            auto& elems = map_entry.second.elements;
            elems.visit(update_element, std::execution::seq);
        }
}

template < el_o_t... orders >
auto partitionMeshImpl(const MeshPartition< orders... >& mesh,
                       idx_t                             n_parts,
                       const util::ArrayOwner< d_id_t >& boundary_ids,
                       util::ArrayOwner< real_t >        part_weights,
                       util::ArrayOwner< idx_t >         node_weights,
                       PartitioningOpts                  opts) -> util::ArrayOwner< MeshPartition< orders... > >
{
    const auto domain_ids  = getDomainIds(mesh, boundary_ids);
    const auto domain_data = getDomainData(mesh, domain_ids);
    auto [epart, npart]    = partitionCondensedMesh(
        mesh, domain_ids, domain_data, n_parts, std::move(part_weights), std::move(node_weights));
    auto new_domain_maps = makeDomainMaps(mesh, n_parts, epart, domain_ids);
    assignBoundaryElements(mesh, epart, new_domain_maps, domain_ids, boundary_ids, domain_data.n_elements);
    auto node_vecs = assignNodes(n_parts, npart, new_domain_maps);
    if (opts.renumber)
        renumberNodes(new_domain_maps, node_vecs);
    return makeMeshFromPartitionComponents(std::move(new_domain_maps), std::move(node_vecs), boundary_ids);
}
} // namespace detail

template < el_o_t... orders, ProblemDef problem_def = EmptyProblemDef{} >
auto partitionMesh(const MeshPartition< orders... >&   mesh,
                   idx_t                               n_parts,
                   util::ArrayOwner< real_t >          part_weights   = {},
                   util::ConstexprValue< problem_def > probdef_ctwrpr = {},
                   PartitioningOpts                    opts = {}) -> util::ArrayOwner< MeshPartition< orders... > >
{
    L3STER_PROFILE_FUNCTION;
    util::throwingAssert(mesh.getGhostNodes().empty() and
                             mesh.getOwnedNodes().back() == mesh.getOwnedNodes().size() - 1,
                         "You cannot partition a mesh which has already been partitioned");
    util::throwingAssert(n_parts >= 1, "The number of resulting partitions cannot be smaller than 1");

    if (n_parts == 1)
    {
        auto retval    = util::ArrayOwner< MeshPartition< orders... > >(1);
        retval.front() = copy(mesh);
        return retval;
    }

    const auto boundary_ids = mesh.getBoundaryIdsCopy();
    auto       node_wgts    = detail::computeNodeWeights(mesh, probdef_ctwrpr);
    return detail::partitionMeshImpl(mesh, n_parts, boundary_ids, std::move(part_weights), std::move(node_wgts), opts);
}
} // namespace lstr::mesh
#endif // L3STER_MESH_PARTITIONMESH_HPP
