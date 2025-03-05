#ifndef L3STER_MESH_LOCALMESHVIEW_HPP
#define L3STER_MESH_LOCALMESHVIEW_HPP

#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/CrsGraph.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/Functional.hpp"
#include "l3ster/util/StaticVector.hpp"
#include "l3ster/util/TbbUtils.hpp"

namespace lstr::mesh
{
template < ElementType ET, el_o_t EO >
class LocalElementView
{
    static constexpr size_t n_nodes           = Element< ET, EO >::n_nodes;
    static constexpr size_t n_internal_nodes  = ElementTraits< Element< ET, EO > >::internal_node_inds.size();
    static constexpr size_t n_boundary_nodes  = ElementTraits< Element< ET, EO > >::boundary_node_inds.size();
    static constexpr bool   optimize_internal = n_internal_nodes > 1;
    static constexpr size_t n_stored_nodes    = optimize_internal ? n_boundary_nodes + 1 : n_nodes;
    static constexpr size_t n_sides           = ElementTraits< Element< ET, EO > >::n_sides;

    using Data  = ElementData< ET, EO >;
    using Nodes = std::array< n_loc_id_t, n_stored_nodes >;
    using Sides = std::array< d_id_t, n_sides >;

    struct BoundaryDescription
    {
        el_side_t side;
        d_id_t    boundary_domain;
    };
    using bound_descr = std::pair< el_side_t, d_id_t >;

public:
    static constexpr auto type  = ET;
    static constexpr auto order = EO;

    template < el_o_t... orders >
    LocalElementView(const Element< ET, EO >&          global_elem,
                     const MeshPartition< orders... >& mesh,
                     std::span< const bound_descr >    sides);

    inline auto getLocalNodes() const -> std::array< n_loc_id_t, n_nodes >
        requires optimize_internal;
    inline auto getLocalNodes() const -> const std::array< n_loc_id_t, n_nodes >& requires(not optimize_internal);
    auto        getData() const -> const Data& { return m_data; }
    inline auto getBoundaries() const -> util::StaticVector< bound_descr, n_sides >;
    [[nodiscard]] bool hasBoundaries() const
        requires(n_sides > 0);
    [[nodiscard]] bool hasBoundaries() const
        requires(n_sides == 0);

private:
    Nodes m_nodes;
    Data  m_data;
    Sides m_sides;
};

template < ElementType ET, el_o_t EO, typename Map >
LocalElementView(const Element< ET, EO >&, const Map&, std::span< const std::pair< el_side_t, d_id_t > >)
    -> LocalElementView< ET, EO >;

template < el_o_t... orders >
struct LocalDomainView
{
    using univec_t = parametrize_type_over_element_types_and_orders_t< util::UniVector, LocalElementView, orders... >;

    univec_t elements;
};

template < ElementType ET, el_o_t EO >
struct LocalElementBoundaryView
{
    static constexpr auto type  = ET;
    static constexpr auto order = EO;

    LocalElementBoundaryView(const LocalElementView< ET, EO >* ptr, el_side_t side) : m_element{ptr}, m_side{side} {}

    auto operator->() const -> const LocalElementView< ET, EO >* { return m_element; }
    auto operator*() const -> const LocalElementView< ET, EO >& { return *m_element; }

    [[nodiscard]] auto        getSide() const { return m_side; }
    [[nodiscard]] inline auto getSideNodeInds() const -> std::span< const el_locind_t >;

private:
    const LocalElementView< ET, EO >* m_element;
    el_side_t                         m_side;
};

template < ElementType ET, el_o_t EO >
auto LocalElementBoundaryView< ET, EO >::getSideNodeInds() const -> std::span< const el_locind_t >
{
    return mesh::getSideNodeIndices< ET, EO >(m_side);
}

/// Locally-indexed view of a mesh
template < el_o_t... orders >
class LocalMeshView
{
public:
    LocalMeshView() = default;
    LocalMeshView(const MeshPartition< orders... >& part, const MeshPartition< orders... >& full);

    template < typename Visitor, SimpleExecutionPolicy_c ExecPolicy = MeshPartition< orders... >::DefaultExec >
    void visit(Visitor&& visitor, ExecPolicy&& policy = {}) const;
    template < typename Visitor, SimpleExecutionPolicy_c ExecPolicy = MeshPartition< orders... >::DefaultExec >
    void visit(Visitor&& visitor, const util::ArrayOwner< d_id_t >& domains, ExecPolicy&& policy = {}) const;
    template < typename Visitor, SimpleExecutionPolicy_c ExecPolicy = MeshPartition< orders... >::DefaultExec >
    void visitBoundaries(Visitor&& visitor, ExecPolicy&& policy = {}) const;
    template < typename Visitor, SimpleExecutionPolicy_c ExecPolicy = MeshPartition< orders... >::DefaultExec >
    void visitBoundaries(Visitor&& visitor, const util::ArrayOwner< d_id_t >& domains, ExecPolicy&& policy = {}) const;

    auto getDomains() const -> const auto& { return m_domains; }
    bool isOwned(n_loc_id_t node) const { return node < m_owned_limit; }
    auto getOwnedNodePredicate() const
    {
        return [this](n_loc_id_t node) {
            return isOwned(node);
        };
    }

    auto getDomainDim(d_id_t domain) const -> std::optional< dim_t >
    {
        const auto iter = m_dims.find(domain);
        if (iter != m_dims.end())
            return {iter->second};
        else
            return std::nullopt;
    }
    template < RangeOfConvertibleTo_c< d_id_t > Domains >
    bool checkDomainDims(Domains&& domains, dim_t dimension_expected) const
    {
        const auto assert_dim = [&](d_id_t domain) {
            const auto dimension_found = getDomainDim(domain);
            return dimension_found.value_or(dimension_expected) == dimension_expected;
        };
        return std::ranges::all_of(std::forward< Domains >(domains), assert_dim);
    }
    dim_t getMaxDim() const { return m_dims.empty() ? invalid_dim : std::ranges::max(m_dims | std::views::values); }

private:
    std::map< d_id_t, LocalDomainView< orders... > > m_domains;
    std::map< d_id_t, dim_t >                        m_dims;
    n_loc_id_t                                       m_owned_limit = 0;
};

template < el_o_t... orders >
template < typename Visitor, SimpleExecutionPolicy_c ExecPolicy >
void LocalMeshView< orders... >::visit(Visitor&&                         visitor,
                                       const util::ArrayOwner< d_id_t >& domains,
                                       ExecPolicy&&                      policy) const
{
    auto       present_ids    = domains | std::views::filter([&](d_id_t d) { return m_domains.contains(d); });
    const auto domain_visitor = [&](d_id_t domain_id) {
        m_domains.at(domain_id).elements.visit(visitor, policy);
    };
    if constexpr (DecaysTo_c< ExecPolicy, std::execution::sequenced_policy >)
        std::ranges::for_each(present_ids, domain_visitor);
    else
    {
        const auto present_ids_rand_acc = util::ArrayOwner{present_ids};
        util::tbb::parallelFor(present_ids_rand_acc, domain_visitor);
    }
}

template < el_o_t... orders >
template < typename Visitor, SimpleExecutionPolicy_c ExecPolicy >
void LocalMeshView< orders... >::visit(Visitor&& visitor, ExecPolicy&& policy) const
{
    visit(std::forward< Visitor >(visitor), m_domains | std::views::keys, std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < typename Visitor, SimpleExecutionPolicy_c ExecPolicy >
void LocalMeshView< orders... >::visitBoundaries(Visitor&&                         visitor,
                                                 const util::ArrayOwner< d_id_t >& domains,
                                                 ExecPolicy&&                      policy) const
{
    const auto dom_set = robin_hood::unordered_flat_set< d_id_t >{domains.begin(), domains.end()};
    const auto in_set  = [&](const auto& pair) {
        return dom_set.contains(pair.second);
    };
    const auto visit_el_boundaries = [&]< ElementType ET, el_o_t EO >(const LocalElementView< ET, EO >& element) {
        if (not element.hasBoundaries())
            return;
        for (auto side : element.getBoundaries() | std::views::filter(in_set) | std::views::keys)
            std::invoke(visitor, LocalElementBoundaryView{&element, side});
    };
    visit(visit_el_boundaries, std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < typename Visitor, SimpleExecutionPolicy_c ExecPolicy >
void LocalMeshView< orders... >::visitBoundaries(Visitor&& visitor, ExecPolicy&& policy) const
{
    const auto visit_el_boundaries = [&]< ElementType ET, el_o_t EO >(const LocalElementView< ET, EO >& element) {
        if (not element.hasBoundaries())
            return;
        for (auto side : element.getBoundaries() | std::views::keys)
            std::invoke(visitor, LocalElementBoundaryView{&element, side});
    };
    visit(visit_el_boundaries, std::forward< ExecPolicy >(policy));
}

template < ElementType ET, el_o_t EO >
template < el_o_t... orders >
LocalElementView< ET, EO >::LocalElementView(const Element< ET, EO >&          global_elem,
                                             const MeshPartition< orders... >& mesh,
                                             std::span< const bound_descr >    sides)
    : m_data{global_elem.getData()}
{
    const auto g2l = [&](auto n) {
        return static_cast< n_loc_id_t >(mesh.getNodeOwnership().getLocalIndex(n));
    };
    if constexpr (optimize_internal)
    {
        std::ranges::transform(getBoundaryNodes(global_elem), m_nodes.begin(), g2l);
        const auto internal_nodes  = getInternalNodes(global_elem) | std::views::transform(g2l);
        const bool internal_sorted = std::ranges::is_sorted(internal_nodes);
        const bool internal_contig = internal_nodes.front() + internal_nodes.size() - 1 == internal_nodes.back();
        util::throwingAssert(internal_sorted and internal_contig, "Unexpected internal node ordering");
        m_nodes.back() = internal_nodes.front();
    }
    else
        std::ranges::transform(global_elem.getNodes(), m_nodes.begin(), g2l);

    m_sides.fill(invalid_domain_id);
    for (const auto& [side, boundary_id] : sides)
        m_sides.at(side) = boundary_id;
}

template < ElementType ET, el_o_t EO >
auto LocalElementView< ET, EO >::getLocalNodes() const -> std::array< n_loc_id_t, n_nodes >
    requires optimize_internal
{
    auto retval = std::array< n_loc_id_t, n_nodes >{};
    for (size_t i = 0; size_t dest_ind : ElementTraits< Element< ET, EO > >::boundary_node_inds)
        retval[dest_ind] = m_nodes[i++];
    for (n_loc_id_t lid = m_nodes.back(); size_t dest_ind : ElementTraits< Element< ET, EO > >::internal_node_inds)
        retval[dest_ind] = lid++;
    return retval;
}

template < ElementType ET, el_o_t EO >
auto LocalElementView< ET, EO >::getLocalNodes() const
    -> const std::array< n_loc_id_t, n_nodes >& requires(not optimize_internal) { return m_nodes; }

template < ElementType ET, el_o_t EO >
auto LocalElementView< ET, EO >::getBoundaries() const -> util::StaticVector< bound_descr, n_sides >
{
    auto retval = util::StaticVector< bound_descr, n_sides >{};
    for (el_side_t side = 0; d_id_t boundary_id : m_sides)
    {
        if (boundary_id != invalid_domain_id)
            retval.push_back({side, boundary_id});
        ++side;
    }
    return retval;
}

template < ElementType ET, el_o_t EO >
bool LocalElementView< ET, EO >::hasBoundaries() const
    requires(n_sides == 0)
{
    return false;
}

template < ElementType ET, el_o_t EO >
bool LocalElementView< ET, EO >::hasBoundaries() const
    requires(n_sides > 0)
{
    return std::ranges::any_of(m_sides, [](d_id_t domain) { return domain != invalid_domain_id; });
}

namespace detail
{
/// Map from nodes to elements which contain them
template < ElementType ET, el_o_t EO >
auto computeNodeToElementMap(const std::vector< LocalElementView< ET, EO > >& elements) -> util::CrsGraph< el_loc_id_t >
{
    using elem_set_t             = robin_hood::unordered_flat_set< el_loc_id_t >;
    auto       node_to_elems_map = robin_hood::unordered_flat_map< n_loc_id_t, elem_set_t >{};
    n_loc_id_t max_node          = 0;
    for (el_loc_id_t elem_id = 0; const auto& elem : elements)
    {
        for (auto node : elem.getLocalNodes())
        {
            node_to_elems_map[node].insert(elem_id);
            max_node = std::max(max_node, node);
        }
        ++elem_id;
    }
    auto node_degs = std::vector< size_t >(max_node + 1);
    for (const auto& [node, elems] : node_to_elems_map)
        node_degs.at(node) = elems.size();
    auto retval = util::CrsGraph< el_loc_id_t >{node_degs};
    for (const auto& [node, elems] : node_to_elems_map)
    {
        const auto row = retval(node);
        std::ranges::copy(elems, row.begin());
        std::ranges::sort(row);
    }
    return retval;
}

/// Model of nodes which are present in cache and how recently they were accessed
class NodeCacheHotnessModel
{
public:
    using hotness_t                    = unsigned;
    static constexpr hotness_t max_hot = 1 << 4;

    void touch(n_loc_id_t node) { m_cache_model[node] = max_hot; }
    void tick()
    {
        m_erase_list.clear();
        for (auto& [node, hot] : m_cache_model)
        {
            hot /= 2;
            if (hot == 0)
                m_erase_list.push_back(node);
        }
        for (auto to_erase : m_erase_list)
            m_cache_model.erase(to_erase);
    }
    auto getActiveNodes() const { return m_cache_model | std::views::all; }

private:
    robin_hood::unordered_flat_map< n_loc_id_t, hotness_t > m_cache_model;
    std::vector< n_loc_id_t >                               m_erase_list;
};

/// Sum of degrees of element nodes in the mesh graph
template < ElementType ET, el_o_t EO >
auto computeElementConnectedness(const std::vector< LocalElementView< ET, EO > >& elements,
                                 const util::CrsGraph< el_loc_id_t >& node_to_elem) -> util::ArrayOwner< unsigned >
{
    auto retval = util::ArrayOwner< unsigned >(elements.size());
    std::ranges::transform(elements, retval.begin(), [&](const auto& elem) {
        const auto& nodes        = elem.getLocalNodes();
        const auto  get_node_deg = [&](auto node) {
            return node_to_elem(node).size();
        };
        return std::transform_reduce(nodes.begin(), nodes.end(), 0u, std::plus{}, get_node_deg);
    });
    return retval;
}

/// Sort vector so that elements sharing nodes are clustered together
template < ElementType ET, el_o_t EO >
void reorderByAdjacency(std::vector< LocalElementView< ET, EO > >& elements, NodeCacheHotnessModel& cache)
{
    const auto nodes_to_elems = computeNodeToElementMap(elements);
    const auto elem_connect   = computeElementConnectedness(elements, nodes_to_elems);
    auto       done_elems     = util::DynamicBitset(elements.size());
    auto       permutation    = std::vector< el_loc_id_t >{};
    permutation.reserve(elements.size());

    auto       hot_elems          = robin_hood::unordered_flat_map< el_loc_id_t, NodeCacheHotnessModel::hotness_t >{};
    const auto populate_hot_elems = [&] {
        for (const auto& [node, hotness] : cache.getActiveNodes())
            for (auto elem : nodes_to_elems(node) | std::views::filter([&](auto e) { return not done_elems.test(e); }))
                hot_elems[elem] += hotness;
    };
    constexpr auto max_unsigned   = std::numeric_limits< unsigned >::max();
    const auto     get_el_connect = [&](el_loc_id_t el) -> unsigned {
        return done_elems.test(el) ? max_unsigned : elem_connect.at(el); // match least connected first
    };
    const auto elem_crit = [&](const auto& map_entry) -> std::array< unsigned, 2 > {
        const auto& [id, hot]  = map_entry;
        const auto neg_connect = max_unsigned - elem_connect.at(id); // match least connected first
        return {hot, neg_connect};
    };
    const auto get_next_elem = [&] {
        if (hot_elems.empty()) [[unlikely]]
            return std::ranges::min(std::views::iota(0u, elements.size()), {}, get_el_connect);
        else
            return std::ranges::max(hot_elems, {}, elem_crit).first;
    };
    const auto put_next_elem = [&](el_loc_id_t elem_id) {
        permutation.push_back(elem_id);
        done_elems.set(elem_id);
        for (auto n : elements.at(elem_id).getLocalNodes())
            cache.touch(n);
    };
    for (size_t i = 0; i != elements.size(); ++i)
    {
        hot_elems.clear();
        populate_hot_elems();
        const auto next = get_next_elem();
        put_next_elem(next);
        cache.tick();
    }

    auto permuted = std::vector< LocalElementView< ET, EO > >{};
    permuted.reserve(elements.size());
    for (auto p : permutation)
        permuted.push_back(elements.at(p));
    elements = std::move(permuted);
}

template < el_o_t... orders >
auto makeElementIdToSidesMap(const MeshPartition< orders... >& mesh)
    -> robin_hood::unordered_flat_map< el_id_t, std::vector< std::pair< el_side_t, d_id_t > > >
{
    auto retval = robin_hood::unordered_flat_map< el_id_t, std::vector< std::pair< el_side_t, d_id_t > > >{};
    for (d_id_t boundary_id : mesh.getBoundaryIdsView())
    {
        const auto& boundary  = mesh.getBoundary(boundary_id);
        const auto  push_side = [&](const auto& bnd_el_view) {
            const auto elem_id = bnd_el_view->getId();
            const auto side    = bnd_el_view.getSide();
            retval[elem_id].emplace_back(side, boundary_id);
        };
        boundary.element_views.visit(push_side, std::execution::seq);
    }
    return retval;
}
} // namespace detail

template < el_o_t... orders >
auto computeNodeOrder(const MeshPartition< orders... >& mesh) -> util::ArrayOwner< n_id_t >
{
    auto       internal_node_set   = robin_hood::unordered_flat_set< n_id_t >{};
    auto       internal_node_vec   = std::vector< n_id_t >{};
    const auto mark_internal_nodes = [&](const auto& element) {
        for (n_id_t node : getInternalNodes(element))
        {
            const auto [_, inserted] = internal_node_set.insert(node);
            util::throwingAssert(inserted, "Internal node present in multiple elements");
            internal_node_vec.push_back(node);
        }
    };
    const dim_t max_dim = mesh.getMaxDim();
    for (d_id_t domain_id : mesh.getDomainIds())
    {
        const auto& [elements, dim] = mesh.getDomain(domain_id);
        if (dim == max_dim)
            elements.visit(mark_internal_nodes, std::execution::seq);
    }
    const auto is_internal = [&](n_id_t node) {
        return internal_node_set.contains(node);
    };
    const auto num_internal = static_cast< size_t >(
        std::ranges::count_if(mesh.getNodeOwnership().owned(), util::negatePredicate(is_internal)));
    const size_t num_total = num_internal + internal_node_vec.size() + mesh.getNodeOwnership().shared().size();
    util::throwingAssert(num_total == mesh.getNNodes(), "Internal nodes cannot be ghost nodes");
    auto retval = util::ArrayOwner< n_id_t >(num_total);
    auto iter =
        std::ranges::copy_if(mesh.getNodeOwnership().owned(), retval.begin(), util::negatePredicate(is_internal)).out;
    iter = std::ranges::copy(internal_node_vec, iter).out;
    std::ranges::copy(mesh.getNodeOwnership().shared(), iter);
    return retval;
}

template < el_o_t... orders >
LocalMeshView< orders... >::LocalMeshView(const MeshPartition< orders... >& part,
                                          const MeshPartition< orders... >& full)
{
    const auto g2l = [&](n_id_t node) {
        return static_cast< n_loc_id_t >(full.getNodeOwnership().getLocalIndex(node));
    };
    const auto owned_nodes      = part.getNodeOwnership().owned();
    m_owned_limit               = owned_nodes.empty() ? 0u : (g2l(owned_nodes.back()) + 1u);
    const auto elem_side_map    = detail::makeElementIdToSidesMap(part);
    const auto element_to_local = [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >& element) {
        const auto side_info = elem_side_map.find(element.getId());
        return side_info != elem_side_map.end() ? LocalElementView{element, full, side_info->second}
                                                : LocalElementView{element, full, {}};
    };
    const auto boundaries = part.getBoundaryIdsCopy();
    auto       cache      = detail::NodeCacheHotnessModel{};
    for (d_id_t domain_id : part.getDomainIds())
    {
        if (std::ranges::binary_search(boundaries, domain_id))
            continue;
        const auto& global_domain = part.getDomain(domain_id);
        auto        local_domain  = LocalDomainView< orders... >{};
        local_domain.elements.reserve(global_domain.elements.sizes());
        const auto make_vec_view = [&]< ElementType ET, el_o_t EO >(const std::vector< Element< ET, EO > >& el_vec) {
            if (el_vec.empty())
                return;
            auto& local_vec = local_domain.elements.template getVector< LocalElementView< ET, EO > >();
            std::ranges::transform(el_vec, std::back_inserter(local_vec), element_to_local);
            detail::reorderByAdjacency(local_vec, cache);
        };
        global_domain.elements.visitVectors(make_vec_view);
        m_domains[domain_id] = std::move(local_domain);
    }
    for (d_id_t dom_id : part.getDomainIds())
        m_dims[dom_id] = part.getDomain(dom_id).dim;
}
} // namespace lstr::mesh
#endif // L3STER_MESH_LOCALMESHVIEW_HPP
