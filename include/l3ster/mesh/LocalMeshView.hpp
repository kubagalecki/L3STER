#ifndef L3STER_MESH_LOCALMESHVIEW_HPP
#define L3STER_MESH_LOCALMESHVIEW_HPP

#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/StaticVector.hpp"

namespace lstr::mesh
{
class NodeMap
{
public:
    inline NodeMap(std::vector< n_id_t > global_nodes);

    n_loc_id_t toLocal(n_id_t gid) const { return m_gid2lid_map.at(gid); }
    n_id_t     toGlobal(n_loc_id_t lid) const { return m_global_nodes.at(lid); }

private:
    std::vector< n_id_t >                                m_global_nodes;
    robin_hood::unordered_flat_map< n_id_t, n_loc_id_t > m_gid2lid_map;
};

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

    using bound_descr = std::pair< el_side_t, d_id_t >;

public:
    static constexpr auto type  = ET;
    static constexpr auto order = EO;

    inline LocalElementView(const Element< ET, EO >&       global_elem,
                            const NodeMap&                 map,
                            std::span< const bound_descr > sides);

    inline auto getLocalNodes() const -> std::array< n_loc_id_t, n_nodes >
        requires optimize_internal;
    inline auto getLocalNodes() const -> const std::array< n_loc_id_t, n_nodes >&
        requires(not optimize_internal);
    inline auto getGlobalNodes(const NodeMap& map) const -> std::array< n_id_t, n_nodes >;
    auto        getData() const -> const Data& { return m_data; }
    inline auto getBoundaries() const -> util::StaticVector< bound_descr, n_sides >;

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

/// Locally-indexed view of a mesh
template < el_o_t... orders >
struct LocalMeshView
{
    explicit LocalMeshView(const MeshPartition< orders... >& mesh);

    std::map< d_id_t, LocalDomainView< orders... > > domains;
    NodeMap                                          node_map;
};

NodeMap::NodeMap(std::vector< n_id_t > global_nodes)
    : m_global_nodes{std::move(global_nodes)}, m_gid2lid_map(m_global_nodes.size())
{
    for (n_loc_id_t lid = 0; n_id_t gid : m_global_nodes)
        m_gid2lid_map.insert({gid, lid++});
}

template < ElementType ET, el_o_t EO >
LocalElementView< ET, EO >::LocalElementView(const Element< ET, EO >&       global_elem,
                                             const NodeMap&                 map,
                                             std::span< const bound_descr > sides)
    : m_data{global_elem.getData()}
{
    if constexpr (optimize_internal)
    {
        const auto g2l = [&](auto n) {
            return map.toLocal(n);
        };
        std::ranges::transform(getBoundaryNodes(global_elem), m_nodes.begin(), g2l);
        const auto internal_nodes  = getInternalNodes(global_elem) | std::views::transform(g2l);
        const bool internal_sorted = std::ranges::is_sorted(internal_nodes);
        const bool internal_contig = internal_nodes.front() + internal_nodes.size() - 1 == internal_nodes.back();
        util::throwingAssert(internal_sorted and internal_contig, "Unexpected internal node ordering");
        m_nodes.back() = internal_nodes.front();
    }
    else
        std::ranges::transform(global_elem.getNodes(), m_nodes.begin(), [&](auto n) { return map.toLocal(n); });

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
auto LocalElementView< ET, EO >::getLocalNodes() const -> const std::array< n_loc_id_t, n_nodes >&
    requires(not optimize_internal)
{
    return m_nodes;
}

template < ElementType ET, el_o_t EO >
auto LocalElementView< ET, EO >::getGlobalNodes(const NodeMap& map) const -> std::array< n_id_t, n_nodes >
{
    auto retval = std::array< n_id_t, n_nodes >{};
    std::ranges::transform(getLocalNodes(), retval.begin(), [&](n_loc_id_t lid) { return map.toGlobal(lid); });
    return retval;
}

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

namespace detail
{
template < el_o_t... orders >
auto computeNodeOrder(const MeshPartition< orders... >& mesh) -> std::vector< n_id_t >
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
    auto retval = std::vector< n_id_t >{};
    retval.reserve(mesh.getAllNodes().size());
    std::ranges::copy_if(mesh.getOwnedNodes(), std::back_inserter(retval), util::negatePredicate(is_internal));
    std::ranges::copy(internal_node_vec, std::back_inserter(retval));
    std::ranges::copy(mesh.getGhostNodes(), std::back_inserter(retval));
    util::throwingAssert(retval.size() == mesh.getAllNodes().size(), "Internal nodes cannot be ghost nodes");
    return retval;
}

template < ElementType ET, el_o_t EO >
auto nodeProjection(const LocalElementView< ET, EO >& element)
{
    return element.getLocalNodes();
}

template < ElementType ET, el_o_t EO >
auto getMaxNode(const std::vector< LocalElementView< ET, EO > >& elements) -> n_loc_id_t
{
    return std::ranges::max(elements | std::views::transform(&nodeProjection< ET, EO >) | std::views::join);
}

/// Sum of degrees of element nodes in the mesh graph
template < ElementType ET, el_o_t EO >
auto computeElementConnectedness(const std::vector< LocalElementView< ET, EO > >& elements, n_loc_id_t max_node)
    -> std::vector< int >
{
    auto node_degs = std::vector< int >(max_node + 1);
    for (auto node : elements | std::views::transform(&nodeProjection< ET, EO >) | std::views::join)
        ++node_degs[node];
    auto retval = std::vector< int >{};
    retval.reserve(elements.size());
    const auto sum_degrees = [&](const LocalElementView< ET, EO >& el) {
        const auto& nodes        = el.getLocalNodes();
        const auto  get_node_deg = [&](auto node) {
            return node_degs.at(node);
        };
        return std::transform_reduce(nodes.begin(), nodes.end(), 0, std::plus{}, get_node_deg);
    };
    std::ranges::transform(elements, std::back_inserter(retval), sum_degrees);
    return retval;
}

class NodeCacheHotness
{
    using hot_t                    = int;
    static constexpr hot_t max_hot = 1 << 5;

public:
    explicit NodeCacheHotness(size_t num_nodes) : m_hot(num_nodes) {}
    hot_t get(n_loc_id_t node) const { return m_hot.at(node); }
    void  touch(n_loc_id_t node) { m_hot.at(node) = max_hot; }
    void  tick()
    {
        for (auto& hot : m_hot)
            hot /= 2; // exponential decay
    }

private:
    std::vector< hot_t > m_hot;
};

/// Sort vector so that elements sharing nodes are clustered together
template < ElementType ET, el_o_t EO >
void reorderByAdjacency(std::vector< LocalElementView< ET, EO > >& elements)
{
    using local_el_t       = std::uint32_t;
    auto       permutation = std::vector< local_el_t >{};
    auto       done        = util::DynamicBitset(elements.size());
    const auto max_node    = getMaxNode(elements);
    const auto el_connect  = computeElementConnectedness(elements, max_node);
    auto       cache       = NodeCacheHotness{max_node + 1};
    const auto elem_crit   = [&](local_el_t el_ind) -> std::pair< int, int > {
        if (done.test(el_ind))
            return {-1, 0}; // Cache hotness is >=0, so no elements get matched twice

        const auto node_hotness = [&](n_loc_id_t node) {
            return cache.get(node);
        };
        const auto& nodes   = elements.at(el_ind).getLocalNodes();
        const auto  hotness = std::transform_reduce(nodes.begin(), nodes.end(), 0, std::plus{}, node_hotness);
        const int   connect = -el_connect.at(el_ind); // We want the least connected elements at the front
        return {hotness, connect};
    };
    const auto put_next_elem = [&](local_el_t elem_id) {
        permutation.push_back(elem_id);
        done.set(elem_id);
        for (auto n : elements.at(elem_id).getLocalNodes())
            cache.touch(n);
    };
    for (size_t i = 0; i != elements.size(); ++i)
    {
        const auto next = std::ranges::max(std::views::iota(0u, elements.size()), {}, elem_crit);
        put_next_elem(next);
        cache.tick();
    }
    auto permuted = std::vector< LocalElementView< ET, EO > >{};
    permuted.reserve(elements.size());
    const auto get_elem = [&](auto i) {
        return elements.at(i);
    };
    std::ranges::transform(std::views::iota(0u, elements.size()), std::back_inserter(permuted), get_elem);
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
LocalMeshView< orders... >::LocalMeshView(const MeshPartition< orders... >& mesh)
    : node_map{detail::computeNodeOrder(mesh)}
{
    const auto elem_side_map    = detail::makeElementIdToSidesMap(mesh);
    const auto element_to_local = [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >& element) {
        const auto side_info = elem_side_map.find(element.getId());
        return side_info != elem_side_map.end() ? LocalElementView{element, node_map, side_info->second}
                                                : LocalElementView{element, node_map, {}};
    };
    const auto boundaries = mesh.getBoundaryIdsCopy();
    for (d_id_t domain_id : mesh.getDomainIds())
    {
        if (std::ranges::binary_search(boundaries, domain_id))
            continue;
        const auto& global_domain = mesh.getDomain(domain_id);
        auto        local_domain  = LocalDomainView< orders... >{};
        local_domain.elements.reserve(global_domain.elements.sizes());
        const auto make_vec_view = [&]< ElementType ET, el_o_t EO >(const std::vector< Element< ET, EO > >& el_vec) {
            if (el_vec.empty())
                return;
            auto& local_vec = local_domain.elements.template getVector< LocalElementView< ET, EO > >();
            std::ranges::transform(el_vec, std::back_inserter(local_vec), element_to_local);
            detail::reorderByAdjacency(local_vec);
        };
        global_domain.elements.visitVectors(make_vec_view);
        domains[domain_id] = std::move(local_domain);
    }
}

} // namespace lstr::mesh
#endif // L3STER_MESH_LOCALMESHVIEW_HPP
