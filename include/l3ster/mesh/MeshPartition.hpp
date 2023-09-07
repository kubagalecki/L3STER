#ifndef L3STER_MESH_MESHPARTITION_HPP
#define L3STER_MESH_MESHPARTITION_HPP

#include "l3ster/mesh/BoundaryView.hpp"
#include "l3ster/mesh/Domain.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/ArrayOwner.hpp"
#include "l3ster/util/MetisUtils.hpp"
#include "l3ster/util/Ranges.hpp"

#include "l3ster/util/RobinHoodHashTables.hpp"

#include <map>

namespace lstr::mesh
{
namespace detail
{
template < el_side_t side, ElementType ET, el_o_t EO, size_t N >
constexpr bool doesSideMatch(const Element< ET, EO >& element, const std::array< n_id_t, N >& sorted_boundary_nodes)
{
    const auto&    side_node_inds = std::get< side >(ElementTraits< Element< ET, EO > >::boundary_table);
    constexpr auto n_side_nodes   = std::tuple_size_v< std::remove_cvref_t< decltype(side_node_inds) > >;
    if constexpr (n_side_nodes == N)
    {
        auto side_nodes = std::array< n_id_t, n_side_nodes >{};
        std::ranges::copy(util::makeIndexedView(element.getNodes(), side_node_inds), side_nodes.begin());
        std::ranges::sort(side_nodes);
        return std::ranges::equal(side_nodes, sorted_boundary_nodes);
    }
    else
        return false;
}

template < ElementType ET, el_o_t EO, size_t N >
constexpr auto matchBoundaryNodesToElement(const Element< ET, EO >&       element,
                                           const std::array< n_id_t, N >& sorted_boundary_nodes)
    -> std::optional< el_side_t >
{
    auto       matched_side   = el_side_t{};
    const auto fold_side_inds = [&]< el_side_t... sides >(std::integer_sequence< el_side_t, sides... >) {
        const auto match_side = [&]< el_side_t side >(std::integral_constant< el_side_t, side >) {
            if (detail::doesSideMatch< side >(element, sorted_boundary_nodes))
            {
                matched_side = side;
                return true;
            }
            else
                return false;
        };
        return (match_side(std::integral_constant< el_side_t, sides >{}) or ...);
    };
    constexpr auto side_inds = std::make_integer_sequence< el_side_t, ElementTraits< Element< ET, EO > >::n_sides >{};
    const auto     matched   = fold_side_inds(side_inds);
    if (matched)
        return {matched_side};
    else
        return std::nullopt;
}
} // namespace detail

template < el_o_t... orders >
class MeshPartition
{
    static_assert(sizeof...(orders) > 0);
    using DefaultExec = const std::execution::sequenced_policy&;

public:
    inline static auto makeBoundaryElementViews(const MeshPartition< orders... >& mesh,
                                                const util::ArrayOwner< d_id_t >& bnd_ids)
        -> detail::boundary_element_view_variant_array_t< orders... >;

private:
    class BoundaryManager
    {
    public:
        BoundaryManager() = default;
        BoundaryManager(const MeshPartition< orders... >& mesh, const util::ArrayOwner< d_id_t >& bnd_ids)
        {
            initBoundaryViews(mesh, bnd_ids);
        }
        void initBoundaryViews(const MeshPartition< orders... >& mesh, const util::ArrayOwner< d_id_t >& bnd_ids)
        {
            m_boundary_views.clear();
            for (d_id_t id : bnd_ids)
            {
                auto bv = BoundaryView< orders... >{makeBoundaryElementViews(mesh, std::views::single(id))};
                m_boundary_views.emplace(id, std::move(bv));
            }
        }

        bool contains(d_id_t id) const { return m_boundary_views.contains(id); }
        auto getBoundaryIdsView() const { return m_boundary_views | std::views::keys; }
        auto getBoundary(d_id_t id) const -> const BoundaryView< orders... >& { return m_boundary_views.at(id); }

    private:
        std::map< d_id_t, BoundaryView< orders... > > m_boundary_views;
    };

public:
    using domain_map_t        = std::map< d_id_t, Domain< orders... > >;
    using find_result_t       = Domain< orders... >::find_result_t;
    using const_find_result_t = Domain< orders... >::const_find_result_t;
    using node_span_t         = std::span< const n_id_t >;

    friend struct SerializedPartition;
    template < el_o_t... O >
    friend auto copy(const MeshPartition< O... >&) -> MeshPartition< O... >;

    MeshPartition() = default;
    inline MeshPartition(domain_map_t domains, const util::ArrayOwner< d_id_t >& boundary_ids);
    MeshPartition(domain_map_t                      domains,
                  util::ArrayOwner< n_id_t >        nodes,
                  size_t                            n_owned_nodes,
                  const util::ArrayOwner< d_id_t >& boundary_ids)
        : m_domains{std::move(domains)},
          m_nodes{std::move(nodes)},
          m_n_owned_nodes{n_owned_nodes},
          m_boundary_manager{*this, boundary_ids}
    {}
    template < SizedRangeOfConvertibleTo_c< n_id_t > Owned, SizedRangeOfConvertibleTo_c< n_id_t > Ghost >
    MeshPartition(domain_map_t                      domains,
                  Owned&&                           nodes,
                  Ghost&&                           ghost_nodes,
                  const util::ArrayOwner< d_id_t >& boundary_ids);

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

    // find
    // Note: if the predicate returns true for multiple elements, it is undefined which one is returned
    template < ElementPredicate_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    auto find(F&& predicate, ExecPolicy&& policy = {}) -> find_result_t;
    template < ElementPredicate_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    auto find(F&& predicate, ExecPolicy&& policy = {}) const -> const_find_result_t;
    template < ElementPredicate_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    auto find(F&& predicate, const util::ArrayOwner< d_id_t >& ids, ExecPolicy&& policy = {}) -> find_result_t;
    template < ElementPredicate_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    auto find(F&& predicate, const util::ArrayOwner< d_id_t >& ids, ExecPolicy&& policy = {}) const
        -> const_find_result_t;
    inline find_result_t       find(el_id_t id);
    inline const_find_result_t find(el_id_t id) const;

    // boundaries
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

    // observers
    auto          getDomainView(d_id_t id) const -> DomainView< orders... > { return DomainView{m_domains.at(id), id}; }
    inline size_t getNElements() const;
    auto          getNDomains() const { return m_domains.size(); }
    auto          getDomainIds() const { return m_domains | std::views::keys; }
    auto          getDomain(d_id_t id) const -> const Domain< orders... >& { return m_domains.at(id); }
    auto          getOwnedNodes() const -> node_span_t { return m_nodes | std::views::take(m_n_owned_nodes); }
    auto          getGhostNodes() const -> node_span_t { return m_nodes | std::views::drop(m_n_owned_nodes); }
    auto          getAllNodes() const -> node_span_t { return m_nodes; }

    inline size_t computeTopoHash() const;
    bool          isGhostNode(n_id_t node) const { return std::ranges::binary_search(getGhostNodes(), node); }
    bool          isOwnedNode(n_id_t node) const { return std::ranges::binary_search(getOwnedNodes(), node); }

    template < el_o_t O >
    auto getConversionAlloc() const -> MeshPartition< O >::domain_map_t;

private:
    inline static auto filterExistingDomainIds(const domain_map_t& domain_map, const util::ArrayOwner< d_id_t >& ids)
        -> util::ArrayOwner< d_id_t >;
    inline static auto filterExistingBoundaryIds(const BoundaryManager&            boundary_manager,
                                                 const util::ArrayOwner< d_id_t >& ids) -> util::ArrayOwner< d_id_t >;
    // Deduce constness based on the domain map, helps with deduplication. Idea similar to C++23 "deducing this"
    template < typename Visitor, typename DomainMap, typename Policy >
    static void
    visitImpl(Visitor&& visitor, DomainMap&& domain_map, const util::ArrayOwner< d_id_t >& domain_ids, Policy&& policy);

    domain_map_t               m_domains;
    util::ArrayOwner< n_id_t > m_nodes;
    size_t                     m_n_owned_nodes;
    BoundaryManager            m_boundary_manager;
};

template < el_o_t... orders >
auto copy(const MeshPartition< orders... >& mesh) -> MeshPartition< orders... >
{
    return {mesh.m_domains, copy(mesh.m_nodes), mesh.m_n_owned_nodes, mesh.getBoundaryIdsCopy()};
}

template < el_o_t... orders >
MeshPartition< orders... >::MeshPartition(MeshPartition< orders... >::domain_map_t domains,
                                          const util::ArrayOwner< d_id_t >&        boundary_ids)
    : m_domains{std::move(domains)}
{
    auto       nodes        = robin_hood::unordered_flat_set< n_id_t >{};
    const auto insert_nodes = [&](const auto& element) {
        for (n_id_t node : element.getNodes())
            nodes.insert(node);
    };
    visit(insert_nodes);
    m_nodes = util::ArrayOwner< n_id_t >(nodes.size());
    std::ranges::copy(nodes, m_nodes.begin());
    std::ranges::sort(m_nodes);
    m_n_owned_nodes = m_nodes.size();

    m_boundary_manager.initBoundaryViews(*this, boundary_ids);
}

template < el_o_t... orders >
template < SizedRangeOfConvertibleTo_c< n_id_t > Owned, SizedRangeOfConvertibleTo_c< n_id_t > Ghost >
MeshPartition< orders... >::MeshPartition(domain_map_t                      domains,
                                          Owned&&                           owned_nodes,
                                          Ghost&&                           ghost_nodes,
                                          const util::ArrayOwner< d_id_t >& boundary_ids)
    : m_domains{std::move(domains)},
      m_nodes(std::ranges::size(owned_nodes) + std::ranges::size(ghost_nodes)),
      m_n_owned_nodes{std::ranges::size(owned_nodes)},
      m_boundary_manager{*this, boundary_ids}
{
    const auto ghost_pos = std::ranges::copy(std::forward< Owned >(owned_nodes), m_nodes.begin()).out;
    std::ranges::copy(std::forward< Ghost >(ghost_nodes), ghost_pos);
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
        m_boundary_manager.getBoundary(id).visit(element_bnd_view_visitor, policy);
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
        return m_domains.at(id).transformReduce(zero, transform, reduction, policy);
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
        return m_boundary_manager.getBoundary(id).transfromReduce(zero, trans, reduction, policy);
    };
    return std::transform_reduce(policy, transred_ids.begin(), transred_ids.end(), zero, reduction, transred_bnd);
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy >
auto MeshPartition< orders... >::find(F&& predicate, ExecPolicy&& policy) -> find_result_t
{
    const auto dom_ids = util::ArrayOwner(getDomainIds());
    return find(std::forward< decltype(predicate) >(predicate), dom_ids, std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy >
auto MeshPartition< orders... >::find(F&& predicate, ExecPolicy&& policy) const -> const_find_result_t
{
    const auto dom_ids = util::ArrayOwner(getDomainIds());
    return find(std::forward< decltype(predicate) >(predicate), dom_ids, std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy >
auto MeshPartition< orders... >::find(F&& predicate, const util::ArrayOwner< d_id_t >& ids, ExecPolicy&& policy)
    -> find_result_t
{
    const auto const_find =
        std::as_const(*this).find(std::forward< F >(predicate), ids, std::forward< ExecPolicy >(policy));
    return detail::makeMutableFindResult< orders... >(const_find);
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy >
auto MeshPartition< orders... >::find(F&& predicate, const util::ArrayOwner< d_id_t >& ids, ExecPolicy&& policy) const
    -> const_find_result_t
{
    auto retval = const_find_result_t{};
    auto mut    = std::mutex{};
    std::find_if(policy, ids.begin(), ids.end(), [&](d_id_t id) {
        const auto map_entry_it = m_domains.find(id);
        if (map_entry_it == m_domains.end())
            return false;
        const auto domain_find_result = map_entry_it->second.find(predicate, policy);
        if (domain_find_result)
        {
            const auto lock = std::lock_guard{mut};
            retval          = domain_find_result;
        }
        return domain_find_result.has_value();
    });
    return retval;
}

template < el_o_t... orders >
auto MeshPartition< orders... >::find(el_id_t id) -> find_result_t
{
    const auto const_find = std::as_const(*this).find(id);
    return detail::makeMutableFindResult< orders... >(const_find);
}

template < el_o_t... orders >
auto MeshPartition< orders... >::find(el_id_t id) const -> const_find_result_t
{
    auto retval = const_find_result_t{};
    std::ranges::find_if(m_domains | std::views::values, [&](const Domain< orders... >& domain) {
        retval = domain.find(id);
        return retval.has_value();
    });
    return retval;
}

template < el_o_t... orders >
size_t MeshPartition< orders... >::getNElements() const
{
    return std::transform_reduce(m_domains.cbegin(), m_domains.cend(), size_t{}, std::plus{}, [](const auto& d) {
        return d.second.getNElements();
    });
}

template < el_o_t... orders >

template < el_o_t O >
auto MeshPartition< orders... >::getConversionAlloc() const -> MeshPartition< O >::domain_map_t
{
    auto retval = typename MeshPartition< O >::domain_map_t{};
    for (const auto& [id, dom] : m_domains)
        retval.emplace_hint(retval.cend(), id, dom.template getConversionAlloc< O >());
    return retval;
}

template < el_o_t... orders >
size_t MeshPartition< orders... >::computeTopoHash() const
{
    constexpr auto hash_range = []< std::ranges::contiguous_range R >(R&& r) -> size_t {
        const auto data = std::span{std::forward< R >(r)};
        return robin_hood::hash_bytes(data.data(), data.size_bytes());
    };
    const auto hash_element = [&](const auto& element) {
        return hash_range(element.getNodes());
    };
    const size_t topo_hash =
        transformReduce(getDomainIds(), size_t{}, hash_element, std::bit_xor<>{}, std::execution::par);
    return topo_hash ^ hash_range(m_nodes);
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
        domain_map.at(id).visit(visitor, policy);
    };
    // std::for_each with sequential policy does not guarantee iteration order
    if constexpr (std::same_as< std::remove_cvref_t< Policy >, std::execution::sequenced_policy >)
        std::ranges::for_each(ids_to_visit, visit_domain_id);
    else
        std::for_each(policy, ids_to_visit.begin(), ids_to_visit.end(), visit_domain_id);
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
        const auto dim = mesh.getDomainView(id).getDim();
        retval[dim].push_back(id);
    }
    return retval;
}
} // namespace detail

template < el_o_t... orders >
auto MeshPartition< orders... >::makeBoundaryElementViews(const MeshPartition< orders... >& mesh,
                                                          const util::ArrayOwner< d_id_t >& bnd_ids)
    -> detail::boundary_element_view_variant_array_t< orders... >
{
    const auto insert_pos_map  = std::invoke([&] {
        auto   retval = robin_hood::unordered_flat_map< el_id_t, size_t >{};
        size_t i      = 0;
        mesh.visit([&](const auto& element) { retval[element.getId()] = i++; }, bnd_ids);
        return retval;
    });
    const auto domain_dim_maps = detail::makeDimToDomainMap(mesh);
    auto       retval          = detail::boundary_element_view_variant_array_t< orders... >(insert_pos_map.size());
    auto       error_flag      = std::atomic_bool{false};
    const auto put_bnd_el_view = [&]< ElementType BET, el_o_t BEO >(const Element< BET, BEO >& bnd_el) {
        const auto bnd_el_nodes_sorted = util::getSortedArray(bnd_el.getNodes());
        const auto match_dom_el        = [&]< ElementType DET, el_o_t DEO >(const Element< DET, DEO >& dom_el) {
            const auto matched_side_opt = detail::matchBoundaryNodesToElement(dom_el, bnd_el_nodes_sorted);
            if (matched_side_opt)
            {
                const auto el_boundary_view = BoundaryElementView{&dom_el, *matched_side_opt};
                const auto insert_pos       = insert_pos_map.at(bnd_el.getId());
                retval[insert_pos]          = el_boundary_view;
            }
            return matched_side_opt.has_value();
        };
        constexpr auto bnd_dim    = ElementTraits< Element< BET, BEO > >::native_dim;
        const auto&    domain_ids = domain_dim_maps.at(bnd_dim + 1);
        const auto     matched    = mesh.find(match_dom_el, domain_ids, std::execution::par);
        if (not matched)
            error_flag.store(true);
    };
    mesh.visit(put_bnd_el_view, bnd_ids, std::execution::par);
    util::throwingAssert(
        not error_flag.load(),
        "BoundaryView could not be constructed because some of the boundary elements are not edges/faces of "
        "any of the domain elements in the partition. This may be because the mesh was partitioned with "
        "incorrectly specified boundaries, resulting in the edge/face element being in a different partition "
        "from its parent area/volume element.");
    return retval;
}
} // namespace lstr::mesh
#endif // L3STER_MESH_MESHPARTITION_HPP
