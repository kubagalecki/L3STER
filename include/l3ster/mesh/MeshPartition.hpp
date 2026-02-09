#ifndef L3STER_MESH_MESHPARTITION_HPP
#define L3STER_MESH_MESHPARTITION_HPP

#include "l3ster/mesh/BoundaryView.hpp"
#include "l3ster/mesh/Domain.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/IndexMap.hpp"
#include "l3ster/util/Ranges.hpp"
#include "l3ster/util/SegmentedOwnership.hpp"

#include <map>
#include <print>

namespace lstr::mesh
{
struct MeshDualGraph
{
    util::CrsGraph< el_loc_id_t >          graph;          // adjacency graph
    util::CrsGraph< unsigned >             weights;        // weights
    std::vector< el_id_t >                 elements;       // element global IDs
    util::IndexMap< el_id_t, el_loc_id_t > els_gid_to_lid; // global-to-local element ID map
};

template < el_o_t... orders >
class MeshPartition
{
    static_assert(sizeof...(orders) > 0);

public:
    using DefaultExec = const std::execution::sequenced_policy&;

    inline static auto makeBoundaryElementViews(const MeshPartition& mesh,
                                                d_id_t               bnd_id,
                                                const MeshDualGraph& dual_graph) -> BoundaryView< orders... >;

private:
    class BoundaryManager
    {
    public:
        BoundaryManager() = default;
        BoundaryManager(const MeshPartition& mesh, const util::ArrayOwner< d_id_t >& bnd_ids)
        {
            initBoundaryViews(mesh, bnd_ids);
        }
        inline void initBoundaryViews(const MeshPartition& mesh, const util::ArrayOwner< d_id_t >& bnd_ids);
        bool        contains(d_id_t id) const { return m_boundary_views.contains(id); }
        auto        getBoundaryIdsView() const { return m_boundary_views | std::views::keys; }
        auto        getBoundary(d_id_t id) const -> const BoundaryView< orders... >& { return m_boundary_views.at(id); }

    private:
        std::map< d_id_t, BoundaryView< orders... > > m_boundary_views;
    };

public:
    using domain_map_t              = std::map< d_id_t, Domain< orders... > >;
    using find_result_t             = Domain< orders... >::find_result_t;
    using const_find_result_t       = Domain< orders... >::const_find_result_t;
    using node_span_t               = std::span< const n_id_t >;
    using node_ownership_t          = util::SegmentedOwnership< n_id_t >;
    using node_ownership_sp_t       = std::shared_ptr< node_ownership_t >;
    using const_node_ownership_sp_t = std::shared_ptr< const node_ownership_t >;

    template < el_o_t... O >
    friend auto copy(const MeshPartition< O... >&) -> MeshPartition< O... >;

    MeshPartition() = default;
    inline MeshPartition(domain_map_t domains, const util::ArrayOwner< d_id_t >& boundary_ids);
    inline MeshPartition(domain_map_t                      domains,
                         n_id_t                            owned_nodes_begin,
                         n_id_t                            n_owned_nodes,
                         const util::ArrayOwner< d_id_t >& boundary_ids);

    MeshPartition(const MeshPartition&)            = delete;
    MeshPartition& operator=(const MeshPartition&) = delete;
    MeshPartition(MeshPartition&&)                 = default;
    MeshPartition& operator=(MeshPartition&&)      = default;
    ~MeshPartition()                               = default;

    // Iteration (visiting) over elements
    template < MutableElementVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    void visit(Visitor&& element_visitor, ExecPolicy&& policy = {});
    template < ConstElementVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    void visit(Visitor&& element_visitor, ExecPolicy&& policy = {}) const;
    template < MutableElementVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    void visit(Visitor&& element_visitor, d_id_t domain_id, ExecPolicy&& policy = {});
    template < ConstElementVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    void visit(Visitor&& element_visitor, d_id_t domain_id, ExecPolicy&& policy = {}) const;
    template < MutableElementVisitor_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    void visit(F&& element_visitor, const util::ArrayOwner< d_id_t >& domain_ids, ExecPolicy&& policy = {});
    template < ConstElementVisitor_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    void visit(F&& element_visitor, const util::ArrayOwner< d_id_t >& domain_ids, ExecPolicy&& policy = {}) const;

    // Reduction over elements
    // Note: `zero` must be the identity element for the reduction (as opposed to, e.g., std::transform_reduce)
    // Note: The iteration order is indeterminate, even if std::execution::seq is passed
    template < std::copy_constructible Zero,
               std::copy_constructible Transform,
               std::copy_constructible Reduction  = std::plus<>,
               SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    auto transformReduce(const util::ArrayOwner< d_id_t >& domain_ids,
                         Zero                              zero,
                         Transform                         transform,
                         Reduction                         reduction = {},
                         ExecPolicy&&                      policy    = {}) const -> Zero
        requires TransformReducible_c< Zero, Transform, Reduction, orders... >;

    // Find
    // Note: if the predicate returns true for multiple elements, it is undefined which one is returned
    template < ElementPredicate_c< orders... > F >
    auto find(F&& predicate) -> find_result_t;
    template < ElementPredicate_c< orders... > F >
    auto find(F&& predicate) const -> const_find_result_t;
    template < ElementPredicate_c< orders... > F >
    auto find(F&& predicate, const util::ArrayOwner< d_id_t >& ids) -> find_result_t;
    template < ElementPredicate_c< orders... > F >
    auto                       find(F&& predicate, const util::ArrayOwner< d_id_t >& ids) const -> const_find_result_t;
    inline find_result_t       find(el_id_t id);
    inline const_find_result_t find(el_id_t id) const;

    // Boundaries
    auto getBoundary(d_id_t id) const -> const BoundaryView< orders... >& { return m_boundary_manager.getBoundary(id); }
    auto getBoundaryIdsView() const { return m_boundary_manager.getBoundaryIdsView(); }
    auto getBoundaryIdsCopy() const -> util::ArrayOwner< d_id_t > { return {getBoundaryIdsView()}; }
    template < BoundaryViewVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    void visitBoundaries(Visitor&&                         element_bnd_view_visitor,
                         const util::ArrayOwner< d_id_t >& boundary_ids,
                         ExecPolicy&&                      policy = {}) const;
    template < std::copy_constructible Zero,
               std::copy_constructible Transform,
               std::copy_constructible Reduction  = std::plus<>,
               ExecutionPolicy_c       ExecPolicy = DefaultExec >
    auto transformReduceBoundaries(const util::ArrayOwner< d_id_t >& boundary_ids,
                                   Zero                              zero,
                                   Transform                         trans,
                                   Reduction                         reduction,
                                   ExecPolicy&&                      policy) const -> Zero
        requires BoundaryTransformReducible_c< Zero, Transform, Reduction, orders... >;

    // Observers
    inline auto getNElements() const -> size_t;
    auto        getNNodes() const -> size_t { return m_node_ownership->localSize(); }
    auto        getNDomains() const { return m_domains.size(); }
    auto        getDomainIds() const { return m_domains | std::views::keys; }
    auto        getDomain(d_id_t id) const -> const Domain< orders... >& { return m_domains.at(id); }
    inline auto getMaxDim() const -> dim_t;
    auto        getNodeOwnership() const -> const node_ownership_t& { return *m_node_ownership; }
    auto        getNodeOwnershipSharedPtr() const -> const_node_ownership_sp_t { return m_node_ownership; }

    // Needed for partitioning
    template < Mapping_c< n_id_t, n_id_t > Map >
    void reindexNodes(const Map& old_to_new);

private:
    inline static auto filterExistingDomainIds(const domain_map_t& domain_map, const util::ArrayOwner< d_id_t >& ids)
        -> util::ArrayOwner< d_id_t >;
    inline static auto filterExistingBoundaryIds(const BoundaryManager&            boundary_manager,
                                                 const util::ArrayOwner< d_id_t >& ids) -> util::ArrayOwner< d_id_t >;
    // Deduce constness based on the domain map, helps with deduplication. Idea similar to C++23 "deducing this"
    template < typename Visitor, typename DomainMap, typename Policy >
    static void
    visitImpl(Visitor&& visitor, DomainMap&& domain_map, const util::ArrayOwner< d_id_t >& domain_ids, Policy&& policy);
    template < typename Predicate, typename DomainMap >
    static auto findImpl(Predicate&& predicate, DomainMap&& domain_map, const util::ArrayOwner< d_id_t >& domain_ids);
    template < typename DomainMap >
    static auto findImpl(el_id_t id, DomainMap&& domain_map);

    auto        ownedBound() const { return m_node_ownership->localSize(); }
    inline void sortElementVectors();

    domain_map_t        m_domains;
    node_ownership_sp_t m_node_ownership;
    BoundaryManager     m_boundary_manager;
};

template < el_o_t... orders >
auto computeMeshDual(const MeshPartition< orders... >& mesh, size_t num_common_nodes) -> MeshDualGraph
{
    L3STER_PROFILE_FUNCTION;
    util::throwingAssert(num_common_nodes > 0);
    auto element_gids = std::vector< el_id_t >{};
    element_gids.reserve(mesh.getNElements());
    mesh.visit([&](const auto& element) { element_gids.push_back(element.id); });
    std::ranges::sort(element_gids);
    auto        element_g2l      = util::IndexMap< el_id_t, el_loc_id_t >{element_gids};
    const auto& node_ownership   = mesh.getNodeOwnership();
    auto        node_degs        = util::ArrayOwner< unsigned >(node_ownership.localSize(), 0);
    const auto  update_node_degs = [&](const auto& element) {
        for (auto n : element.nodes)
            std::atomic_ref{node_degs[node_ownership.getLocalIndex(n)]}.fetch_add(1, std::memory_order_relaxed);
    };
    mesh.visit(update_node_degs, std::execution::par);
    auto       node2elems     = util::CrsGraph< el_loc_id_t >{node_degs};
    const auto write_elem_ids = [&](const auto& element) {
        const auto elid = element_g2l(element.id);
        for (auto n : element.nodes)
        {
            const auto nlid         = node_ownership.getLocalIndex(n);
            const auto index        = std::atomic_ref{node_degs[nlid]}.fetch_sub(1, std::memory_order_acq_rel) - 1u;
            node2elems(nlid)[index] = elid;
        }
    };
    mesh.visit(write_elem_ids, std::execution::par);
    auto       elem_degs               = util::ArrayOwner< unsigned >(element_gids.size(), 0);
    const auto get_neighbors_with_reps = [&](const auto& element, el_loc_id_t elid) {
        auto retval = element.nodes |
                      std::views::transform([&](auto node) { return node2elems(node_ownership.getLocalIndex(node)); }) |
                      std::views::join | std::views::filter(std::bind_back(std::not_equal_to{}, elid)) |
                      std::ranges::to< util::ArrayOwner >();
        std::ranges::sort(retval);
        return retval;
    };
    const auto update_elem_degs = [&](const auto& element) {
        const auto elid     = element_g2l(element.id);
        const auto nbrs     = get_neighbors_with_reps(element, elid);
        const auto num_nbrs = std::ranges::count_if(nbrs | std::views::chunk_by(std::equal_to{}), [&](auto&& reps) {
            return std::ranges::size(reps) >= num_common_nodes;
        });
        elem_degs.at(elid)  = static_cast< unsigned >(num_nbrs);
    };
    mesh.visit(update_elem_degs, std::execution::par);
    auto       dual_graph       = util::CrsGraph< el_loc_id_t >{elem_degs};
    auto       weights          = util::CrsGraph< unsigned >{elem_degs};
    const auto write_graph_data = [&](const auto& element) {
        const auto elid        = element_g2l(element.id);
        const auto nbrs        = get_neighbors_with_reps(element, elid);
        const auto dest_verts  = dual_graph(elid);
        const auto dest_wgts   = weights(elid);
        auto       reps_chunks = nbrs | std::views::chunk_by(std::equal_to{}) |
                           std::views::filter([&](auto&& r) { return std::ranges::size(r) >= num_common_nodes; });
        for (auto&& [v, w, reps] : std::views::zip(dest_verts, dest_wgts, reps_chunks))
        {
            v = *std::ranges::begin(reps);
            w = static_cast< unsigned >(std::ranges::size(reps));
        }
    };
    mesh.visit(write_graph_data, std::execution::par);
    return {std::move(dual_graph), std::move(weights), std::move(element_gids), std::move(element_g2l)};
}

template < el_o_t... orders >
template < Mapping_c< n_id_t, n_id_t > Map >
void MeshPartition< orders... >::reindexNodes(const Map& old_to_new)
{
    const auto o2n             = std::cref(old_to_new);
    const auto old_owned       = m_node_ownership->owned();
    const auto num_owned       = old_owned.size();
    const auto new_begin       = old_owned.empty() ? 0uz : std::ranges::min(old_owned | std::views::transform(o2n));
    *m_node_ownership          = {new_begin, num_owned, m_node_ownership->shared() | std::views::transform(o2n)};
    const auto reindex_element = [&]< ElementType ET, el_o_t EO >(Element< ET, EO >& element) {
        for (auto& node : element.nodes)
            node = std::invoke(old_to_new, node);
    };
    visit(reindex_element, std::execution::par);
}

template < el_o_t... orders >
void MeshPartition< orders... >::BoundaryManager::initBoundaryViews(const MeshPartition&              mesh,
                                                                    const util::ArrayOwner< d_id_t >& bnd_ids)
{
    L3STER_PROFILE_FUNCTION;
    m_boundary_views.clear();
    const auto dual_graph = computeMeshDual(mesh, 2);
    for (auto id : bnd_ids)
    {
        auto bv = makeBoundaryElementViews(mesh, id, dual_graph);
        m_boundary_views.emplace(id, std::move(bv));
    }
}

template < el_o_t... orders >
auto MeshPartition< orders... >::getMaxDim() const -> dim_t
{
    if (m_domains.empty())
        return Domain< orders... >::uninitialized_dim;
    return std::ranges::max(m_domains | std::views::transform([](const auto& pair) { return pair.second.dim; }));
}

template < el_o_t... orders >
template < typename Predicate, typename DomainMap >
auto MeshPartition< orders... >::findImpl(Predicate&&                       predicate,
                                          DomainMap&&                       domain_map,
                                          const util::ArrayOwner< d_id_t >& domain_ids)
{
    for (auto id : filterExistingDomainIds(domain_map, domain_ids))
    {
        auto&&     domain   = domain_map.at(id);
        const auto find_res = domain.elements.find(predicate);
        if (find_res)
            return find_res;
    }
    using retval_t = decltype(domain_map.at(0).elements.find(predicate));
    return retval_t{};
}

template < el_o_t... orders >
template < typename DomainMap >
auto MeshPartition< orders... >::findImpl(el_id_t id, DomainMap&& domain_map)
{
    using retval_t            = decltype(domain_map.at(0).elements.find([](const auto&) { return true; }));
    auto       retval         = retval_t{};
    const auto find_in_domain = [&](auto&& domain) {
        const auto find_in_vec = [&](auto&& el_vec) {
            if (el_vec.empty())
                return false;

            const auto front_id = el_vec.front().id;
            const auto back_id  = el_vec.back().id;

            if (id < front_id or id > back_id)
                return false;

            // Optimization for contiguous case
            if (back_id - front_id + 1u == el_vec.size())
            {
                const auto ptr = std::addressof(el_vec[id - front_id]);
                retval.emplace(ptr);
                return true;
            }

            const auto iter = std::ranges::lower_bound(el_vec, id, {}, [](const auto& el) { return el.id; });
            if (iter == end(el_vec) or iter->id != id)
                return false;

            const auto ptr = std::addressof(*iter);
            retval.emplace(ptr);
            return true;
        };
        domain.elements.visitVectorsUntil(find_in_vec);
        return retval.has_value();
    };
    std::ranges::find_if(domain_map | std::views::values, find_in_domain);
    return retval;
}

template < el_o_t... orders >
void MeshPartition< orders... >::sortElementVectors()
{
    constexpr auto sort_elvec = []< ElementType T, el_o_t O >(std::vector< Element< T, O > >& v) {
        std::ranges::sort(v, {}, &Element< T, O >::id);
    };
    for (auto& [_, domain] : m_domains)
        domain.elements.visitVectors(sort_elvec);
}

template < el_o_t... orders >
auto copy(const MeshPartition< orders... >& mesh) -> MeshPartition< orders... >
{
    return {mesh.m_domains,
            mesh.getNodeOwnership().owned().size() == 0 ? 0uz : mesh.getNodeOwnership().owned().front(),
            mesh.getNodeOwnership().owned().size(),
            mesh.getBoundaryIdsCopy()};
}

template < el_o_t... orders >
MeshPartition< orders... >::MeshPartition(domain_map_t domains, const util::ArrayOwner< d_id_t >& boundary_ids)
    : m_domains{std::move(domains)}
{
    sortElementVectors();

    auto       nodes        = robin_hood::unordered_flat_set< n_id_t >{};
    const auto insert_nodes = [&](const auto& element) {
        for (n_id_t node : element.nodes)
            nodes.insert(node);
    };
    visit(insert_nodes);
    const auto num_owned   = nodes.size();
    const auto owned_begin = nodes.empty() ? 0uz : std::ranges::min(nodes);
    m_node_ownership       = std::make_shared< node_ownership_t >(owned_begin, num_owned, std::views::empty< n_id_t >);
    m_boundary_manager.initBoundaryViews(*this, boundary_ids);
}

template < el_o_t... orders >
MeshPartition< orders... >::MeshPartition(domain_map_t                      domains,
                                          n_id_t                            owned_nodes_begin,
                                          n_id_t                            n_owned_nodes,
                                          const util::ArrayOwner< d_id_t >& boundary_ids)
    : m_domains{std::move(domains)}
{
    sortElementVectors();

    const auto owned_bound = owned_nodes_begin + n_owned_nodes;
    auto       ghost_set   = robin_hood::unordered_flat_set< n_id_t >{};
    visit([&](const auto& el) {
        for (auto n : el.nodes)
            if (n < owned_nodes_begin or n >= owned_bound)
                ghost_set.insert(n);
    });
    m_node_ownership   = std::make_shared< node_ownership_t >(owned_nodes_begin, n_owned_nodes, ghost_set);
    m_boundary_manager = BoundaryManager{*this, boundary_ids};
}

// Implementation note: It should be possible to sequentially traverse elements in a deterministic order (e.g. the mesh
// partitioning facilities rely on this). However, std::execution::sequenced_policy does not make this guarantee (it
// only guarantees sequential execution). For this reason, the code below contains compile-time conditionals which
// ensure that the "classic" (i.e. deterministically sequenced) traversal algorithms is called when std::execution::seq
// is passed to the member function.
template < el_o_t... orders >
template < MutableElementVisitor_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition< orders... >::visit(F&& element_visitor, d_id_t domain_id, ExecPolicy&& policy)
{
    visit(std::forward< decltype(element_visitor) >(element_visitor),
          std::views::single(domain_id),
          std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < ConstElementVisitor_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition< orders... >::visit(F&& element_visitor, d_id_t domain_id, ExecPolicy&& policy) const
{
    visit(std::forward< decltype(element_visitor) >(element_visitor),
          std::views::single(domain_id),
          std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < MutableElementVisitor_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition< orders... >::visit(F&& element_visitor, ExecPolicy&& policy)
{
    visitImpl(std::forward< F >(element_visitor), m_domains, getDomainIds(), std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < ConstElementVisitor_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition< orders... >::visit(F&& element_visitor, ExecPolicy&& policy) const
{
    visitImpl(std::forward< F >(element_visitor), m_domains, getDomainIds(), std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < MutableElementVisitor_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition< orders... >::visit(F&&                               element_visitor,
                                       const util::ArrayOwner< d_id_t >& domain_ids,
                                       ExecPolicy&&                      policy)
{
    visitImpl(std::forward< F >(element_visitor), m_domains, domain_ids, std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < ConstElementVisitor_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition< orders... >::visit(F&&                               element_visitor,
                                       const util::ArrayOwner< d_id_t >& domain_ids,
                                       ExecPolicy&&                      policy) const
{
    visitImpl(std::forward< F >(element_visitor), m_domains, domain_ids, std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < BoundaryViewVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition< orders... >::visitBoundaries(Visitor&&                         element_bnd_view_visitor,
                                                 const util::ArrayOwner< d_id_t >& boundary_ids,
                                                 ExecPolicy&&                      policy) const
{
    const auto visit_ids      = filterExistingBoundaryIds(m_boundary_manager, boundary_ids);
    const auto visit_bnd_view = [&](d_id_t id) {
        m_boundary_manager.getBoundary(id).element_views.visit(element_bnd_view_visitor, policy);
    };
    std::for_each(policy, visit_ids.begin(), visit_ids.end(), visit_bnd_view);
}

template < el_o_t... orders >
template < std::copy_constructible Zero,
           std::copy_constructible Transform,
           std::copy_constructible Reduction,
           SimpleExecutionPolicy_c ExecPolicy >
auto MeshPartition< orders... >::transformReduce(const util::ArrayOwner< d_id_t >& domain_ids,
                                                 Zero                              zero,
                                                 Transform                         transform,
                                                 Reduction                         reduction,
                                                 ExecPolicy&&                      policy) const -> Zero
    requires TransformReducible_c< Zero, Transform, Reduction, orders... >
{
    const auto ids           = filterExistingDomainIds(m_domains, domain_ids);
    const auto reduce_domain = [&](d_id_t id) {
        return m_domains.at(id).elements.transformReduce(zero, transform, reduction, policy);
    };
    return std::transform_reduce(policy, ids.begin(), ids.end(), std::move(zero), std::move(reduction), reduce_domain);
}

template < el_o_t... orders >
template < std::copy_constructible Zero,
           std::copy_constructible Transform,
           std::copy_constructible Reduction,
           ExecutionPolicy_c       ExecPolicy >
auto MeshPartition< orders... >::transformReduceBoundaries(const util::ArrayOwner< d_id_t >& boundary_ids,
                                                           Zero                              zero,
                                                           Transform                         trans,
                                                           Reduction                         reduction,
                                                           ExecPolicy&&                      policy) const -> Zero
    requires BoundaryTransformReducible_c< Zero, Transform, Reduction, orders... >
{
    const auto transred_ids = filterExistingBoundaryIds(m_boundary_manager, boundary_ids);
    const auto transred_bnd = [&](d_id_t id) {
        return m_boundary_manager.getBoundary(id).element_views.transformReduce(zero, trans, reduction, policy);
    };
    return std::transform_reduce(policy, transred_ids.begin(), transred_ids.end(), zero, reduction, transred_bnd);
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > F >
auto MeshPartition< orders... >::find(F&& predicate) -> find_result_t
{
    return findImpl(std::forward< F >(predicate), m_domains, getDomainIds());
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > F >
auto MeshPartition< orders... >::find(F&& predicate) const -> const_find_result_t
{
    return findImpl(std::forward< F >(predicate), m_domains, getDomainIds());
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > F >
auto MeshPartition< orders... >::find(F&& predicate, const util::ArrayOwner< d_id_t >& ids) -> find_result_t
{
    return findImpl(std::forward< F >(predicate), m_domains, ids);
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > F >
auto MeshPartition< orders... >::find(F&& predicate, const util::ArrayOwner< d_id_t >& ids) const -> const_find_result_t
{
    return findImpl(std::forward< F >(predicate), m_domains, ids);
}

template < el_o_t... orders >
auto MeshPartition< orders... >::find(el_id_t id) -> find_result_t
{
    return findImpl(id, m_domains);
}

template < el_o_t... orders >
auto MeshPartition< orders... >::find(el_id_t id) const -> const_find_result_t
{
    return findImpl(id, m_domains);
}

template < el_o_t... orders >
auto MeshPartition< orders... >::getNElements() const -> size_t
{
    return std::transform_reduce(
        m_domains.cbegin(), m_domains.cend(), 0uz, std::plus{}, [](const auto& d) { return d.second.elements.size(); });
}

template < el_o_t... orders >
template < typename Visitor, typename DomainMap, typename Policy >
void MeshPartition< orders... >::visitImpl(Visitor&&                         visitor,
                                           DomainMap&&                       domain_map,
                                           const util::ArrayOwner< d_id_t >& domain_ids,
                                           Policy&&                          policy)
{
    const auto ids_to_visit    = filterExistingDomainIds(domain_map, domain_ids);
    const auto visit_domain_id = [&](d_id_t id) {
        domain_map.at(id).elements.visit(visitor, policy);
    };
    // std::for_each with sequential policy does not guarantee iteration order
    if constexpr (std::same_as< std::remove_cvref_t< Policy >, std::execution::sequenced_policy >)
        std::ranges::for_each(ids_to_visit, visit_domain_id);
    else
        util::tbb::parallelFor(ids_to_visit, visit_domain_id);
}

template < el_o_t... orders >
auto MeshPartition< orders... >::filterExistingDomainIds(const domain_map_t&               domain_map,
                                                         const util::ArrayOwner< d_id_t >& ids)
    -> util::ArrayOwner< d_id_t >
{
    return ids | std::views::filter([&](d_id_t id) { return domain_map.contains(id); });
}

template < el_o_t... orders >
auto MeshPartition< orders... >::filterExistingBoundaryIds(const BoundaryManager&            boundary_manager,
                                                           const util::ArrayOwner< d_id_t >& ids)
    -> util::ArrayOwner< d_id_t >
{
    return ids | std::views::filter([&](d_id_t id) { return boundary_manager.contains(id); });
}

namespace detail
{
template < el_o_t... orders >
auto makeDimToDomainMap(const MeshPartition< orders... >& mesh) -> std::map< dim_t, std::vector< d_id_t > >
{
    auto retval = std::map< dim_t, std::vector< d_id_t > >{};
    for (d_id_t id : mesh.getDomainIds())
    {
        const auto dim = mesh.getDomain(id).dim;
        retval[dim].push_back(id);
    }
    return retval;
}
} // namespace detail

template < el_o_t... orders >
auto MeshPartition< orders... >::makeBoundaryElementViews(const MeshPartition& mesh,
                                                          d_id_t               bnd_id,
                                                          const MeshDualGraph& dual_graph) -> BoundaryView< orders... >
{
    if (not mesh.m_domains.contains(bnd_id))
        return {};

    constexpr std::string_view not_found_error =
        "BoundaryView could not be constructed because some of the boundary elements are not edges/faces of "
        "any of the domain elements in the partition. This may be because the mesh was partitioned with "
        "incorrectly specified boundaries, resulting in the edge/face element being in a different partition "
        "from its parent area/volume element.";

    const auto domain_dim_maps = detail::makeDimToDomainMap(mesh);
    const auto boundary_dim    = mesh.m_domains.at(bnd_id).dim;
    util::throwingAssert(domain_dim_maps.contains(boundary_dim + 1), not_found_error);

    auto el_ids    = std::vector< el_id_t >{};
    auto num_nodes = std::vector< unsigned >{};
    el_ids.reserve(mesh.getDomain(bnd_id).numElements());
    num_nodes.reserve(el_ids.capacity());
    const auto push_el_data = [&](const auto& element) {
        el_ids.push_back(element.id);
        num_nodes.push_back(static_cast< unsigned >(element.nodes.size()));
    };
    mesh.visit(push_el_data, {bnd_id});
    auto   nodes = util::CrsGraph< n_id_t >{num_nodes};
    size_t eli   = 0;
    mesh.visit([&](const auto& element) { std::ranges::copy(element.nodes, nodes(eli++).begin()); }, {bnd_id});

    const auto& [graph, wgts, gids, gid2lid] = dual_graph;
    auto       not_found_flag                = std::atomic_bool{false};
    const auto get_parent_side = [&](el_id_t egid, std::span< n_id_t > el_nodes) -> std::pair< el_id_t, el_side_t > {
        const auto elid           = gid2lid(egid);
        const auto nbrs_lids      = graph(elid);
        const auto n_common       = wgts(elid);
        const auto parent_wgt_ind = std::ranges::find_if(n_common, std::bind_back(std::equal_to{}, el_nodes.size()));
        if (parent_wgt_ind == n_common.end()) [[unlikely]]
        {
            not_found_flag.store(true, std::memory_order_relaxed);
            return {};
        }
        const auto parent_ind     = static_cast< size_t >(std::distance(n_common.begin(), parent_wgt_ind));
        const auto parent_lid     = nbrs_lids[parent_ind];
        const auto parent_gid     = gids.at(parent_lid);
        const auto parent_element = mesh.find(parent_gid);
        std::ranges::sort(el_nodes);
        const auto compute_side = [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >* element_ptr) {
            constexpr auto num_sides  = ElementTraits< Element< ET, EO > >::n_sides;
            constexpr auto side_range = std::views::iota(el_side_t{}, num_sides);
            const auto     side_iter  = std::ranges::find_if(side_range, [&](auto side) {
                const auto bnd_view   = BoundaryElementView< ET, EO >{element_ptr, side};
                auto       side_nodes = bnd_view.getSideNodesView() | std::ranges::to< util::ArrayOwner >();
                std::ranges::sort(side_nodes);
                return std::ranges::equal(el_nodes, side_nodes);
            });
            util::throwingAssert(side_iter != side_range.end());
            return *side_iter;
        };
        return {parent_gid, std::visit(compute_side, parent_element.value())};
    };
    const auto el_nodes =
        std::views::iota(0uz, el_ids.size()) | std::views::transform([&](auto i) { return nodes(i); });
    auto parents_with_sides = util::ArrayOwner< std::pair< el_id_t, el_side_t > >(el_ids.size());
    util::tbb::parallelTransform(
        std::views::zip_transform(get_parent_side, el_ids, el_nodes), parents_with_sides.begin(), std::identity{});
    // std::ranges::copy(std::views::zip_transform(get_parent_side, el_ids, el_nodes), parents_with_sides.begin());
    util::throwingAssert(not not_found_flag.load(), not_found_error);

    auto retval = BoundaryView< orders... >{};
    for (const auto& [el_id, side] : parents_with_sides)
    {
        const auto parent_element       = mesh.find(el_id);
        const auto emplace_element_view = [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >* ptr) {
            retval.element_views.template getVector< BoundaryElementView< ET, EO > >().emplace_back(ptr, side);
        };
        std::visit(emplace_element_view, parent_element.value());
    }
    return retval;
}
} // namespace lstr::mesh
#endif // L3STER_MESH_MESHPARTITION_HPP
