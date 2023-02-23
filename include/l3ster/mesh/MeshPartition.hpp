#ifndef L3STER_MESH_MESHPARTITION_HPP
#define L3STER_MESH_MESHPARTITION_HPP

#include "l3ster/mesh/BoundaryView.hpp"
#include "l3ster/mesh/Domain.hpp"
#include "l3ster/mesh/ElementSideMatching.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/MetisUtils.hpp"

#include "l3ster/util/RobinHoodHashTables.hpp"

#include <map>

namespace lstr
{
namespace detail
{
template < typename T >
concept DomainPredicate_c = requires(T op, const DomainView dv) {
                                {
                                    std::invoke(op, dv)
                                } -> std::convertible_to< bool >;
                            };
template < typename T >
concept DomainIdRange_c = SizedRangeOfConvertibleTo_c< T, d_id_t >;
} // namespace detail

class MeshPartition
{
public:
    using domain_map_t              = std::map< d_id_t, Domain >;
    using node_vec_t                = std::vector< n_id_t >;
    using find_result_t             = std::optional< std::pair< element_ptr_variant_t, d_id_t > >;
    using cfind_result_t            = std::optional< std::pair< element_cptr_variant_t, d_id_t > >;
    using el_boundary_view_result_t = std::pair< cfind_result_t, el_side_t >;
    using node_span_t               = std::span< const n_id_t >;

    friend struct SerializedPartition;

    MeshPartition() = default;
    inline explicit MeshPartition(domain_map_t domains);
    MeshPartition(domain_map_t domains, std::vector< n_id_t > nodes, size_t n_owned_nodes)
        : m_domains{std::move(domains)}, m_nodes{std::move(nodes)}, m_n_owned_nodes{n_owned_nodes}
    {}
    MeshPartition(domain_map_t                                 domains,
                  SizedRangeOfConvertibleTo_c< n_id_t > auto&& nodes,
                  SizedRangeOfConvertibleTo_c< n_id_t > auto&& ghost_nodes);

    inline const MetisGraphWrapper& initDualGraph();
    void                            deleteDualGraph() noexcept { m_dual_graph.reset(); }
    [[nodiscard]] bool              isDualGraphInitialized() const noexcept { return m_dual_graph.has_value(); }
    [[nodiscard]] inline const MetisGraphWrapper& getDualGraph() const;

    // Iteration (visiting) over elements
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_elements auto&& element_visitor, d_id_t domain_id, ExecPolicy&& policy = {});
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_const_elements auto&& element_visitor, d_id_t domain_id, ExecPolicy&& policy = {}) const;
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_elements auto&&     element_visitor,
               detail::DomainPredicate_c auto&& domain_predicate,
               ExecPolicy&&                     policy = {});
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_const_elements auto&& element_visitor,
               detail::DomainPredicate_c auto&&   domain_predicate,
               ExecPolicy&&                       policy = {}) const;
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_elements_and< const DomainView > auto&& element_visitor,
               detail::DomainPredicate_c auto&&                     domain_predicate,
               ExecPolicy&&                                         policy = {});
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_const_elements_and< const DomainView > auto&& element_visitor,
               detail::DomainPredicate_c auto&&                           domain_predicate,
               ExecPolicy&&                                               policy = {}) const;
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_elements auto&& element_visitor, ExecPolicy&& policy = {});
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_const_elements auto&& element_visitor, ExecPolicy&& policy = {}) const;
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_elements_and< const DomainView > auto&& element_visitor, ExecPolicy&& policy = {});
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_const_elements_and< const DomainView > auto&& element_visitor,
               ExecPolicy&&                                               policy = {}) const;
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_elements auto&&   element_visitor,
               detail::DomainIdRange_c auto&& domain_ids,
               ExecPolicy&&                   policy = {});
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_const_elements auto&& element_visitor,
               detail::DomainIdRange_c auto&&     domain_ids,
               ExecPolicy&&                       policy = {}) const;
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_elements_and< const DomainView > auto&& element_visitor,
               detail::DomainIdRange_c auto&&                       domain_ids,
               ExecPolicy&&                                         policy = {});
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(invocable_on_const_elements_and< const DomainView > auto&& element_visitor,
               detail::DomainIdRange_c auto&&                             domain_ids,
               ExecPolicy&&                                               policy = {}) const;

    // Reduction over elements
    // Note: `zero` must be the identity element for the reduction (as opposed to, e.g., std::transform_reduce)
    // Note: The iteration order is indeterminate, even if std::execution::seq is passed
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    auto transformReduce(const auto&                    zero,
                         auto&&                         reduction,
                         auto&&                         transform,
                         detail::DomainIdRange_c auto&& domain_ids,
                         ExecPolicy&&                   policy = {}) const -> std::decay_t< decltype(zero) >
        requires invocable_on_const_elements_r< decltype(transform), std::decay_t< decltype(zero) > > and
                 requires {
                     {
                         std::invoke(reduction, zero, zero)
                     } -> std::convertible_to< std::decay_t< decltype(zero) > >;
                 };

    // find
    // Note: if the predicate returns true for multiple elements, the reference to any one of them may be returned
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    [[nodiscard]] find_result_t find(invocable_on_const_elements_r< bool > auto&& predicate, ExecPolicy&& policy = {});
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    [[nodiscard]] cfind_result_t find(invocable_on_const_elements_r< bool > auto&& predicate,
                                      ExecPolicy&&                                 policy = {}) const;
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    [[nodiscard]] find_result_t find(invocable_on_const_elements_r< bool > auto&& predicate,
                                     detail::DomainPredicate_c auto&&             domain_predicate,
                                     ExecPolicy&&                                 policy = {});
    template < SimpleExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    [[nodiscard]] cfind_result_t        find(invocable_on_const_elements_r< bool > auto&& predicate,
                                             detail::DomainPredicate_c auto&&             domain_ids,
                                             ExecPolicy&&                                 policy = {}) const;
    [[nodiscard]] inline find_result_t  find(el_id_t id);
    [[nodiscard]] inline cfind_result_t find(el_id_t id) const;

    // boundary views
    template < ElementTypes T, el_o_t O >
    el_boundary_view_result_t  getElementBoundaryView(const Element< T, O >& el, d_id_t d) const;
    [[nodiscard]] BoundaryView getBoundaryView(detail::DomainIdRange_c auto&& boundary_ids) const;
    [[nodiscard]] BoundaryView getBoundaryView(d_id_t id) const { return getBoundaryView(std::views::single(id)); }

    // observers
    [[nodiscard]] DomainView    getDomainView(d_id_t id) const { return DomainView{m_domains.at(id), id}; }
    [[nodiscard]] inline size_t getNElements() const;
    [[nodiscard]] auto          getNDomains() const { return m_domains.size(); }
    [[nodiscard]] auto          getDomainIds() const { return m_domains | std::views::keys; }
    [[nodiscard]] const Domain& getDomain(d_id_t id) const { return m_domains.at(id); }
    [[nodiscard]] node_span_t   getOwnedNodes() const { return m_nodes | std::views::take(m_n_owned_nodes); }
    [[nodiscard]] node_span_t   getGhostNodes() const { return m_nodes | std::views::drop(m_n_owned_nodes); }
    [[nodiscard]] node_span_t   getAllNodes() const { return m_nodes; }

    [[nodiscard]] inline size_t computeTopoHash() const;
    [[nodiscard]] bool isGhostNode(n_id_t node) const { return std::ranges::binary_search(getGhostNodes(), node); }
    [[nodiscard]] bool isOwnedNode(n_id_t node) const { return std::ranges::binary_search(getOwnedNodes(), node); }

    template < el_o_t O >
    [[nodiscard]] domain_map_t getConversionAlloc() const;

private:
    [[nodiscard]] inline auto              convertToMetisFormat() const;
    [[nodiscard]] inline MetisGraphWrapper makeMetisDualGraph() const;

    static inline cfind_result_t constifyFindResult(find_result_t found);

    static auto filterExistingDomainIds(const auto& domain_map, detail::DomainIdRange_c auto&& ids)
        -> std::vector< d_id_t >;
    // Deduce constness based on the domain map, helps with deduplication. Idea similar to C++23 "deducing this"
    static void visitImpl(auto&& visitor, auto&& domain_map, auto&& domain_ids, SimpleExecutionPolicy_c auto&& policy);

    template < ElementTypes T, el_o_t O >
    el_boundary_view_result_t getElementBoundaryViewImpl(const Element< T, O >& el) const;
    template < ElementTypes T, el_o_t O >
    el_boundary_view_result_t getElementBoundaryViewFallback(const Element< T, O >& el, d_id_t d) const;

    domain_map_t                       m_domains;
    node_vec_t                         m_nodes;
    size_t                             m_n_owned_nodes;
    std::optional< MetisGraphWrapper > m_dual_graph;
};

MeshPartition::MeshPartition(MeshPartition::domain_map_t domains_) : m_domains{std::move(domains_)}
{
    robin_hood::unordered_flat_set< n_id_t > nodes;
    visit([&](const auto& element) { std::ranges::for_each(element.getNodes(), [&](n_id_t n) { nodes.insert(n); }); });
    m_nodes.reserve(nodes.size());
    std::ranges::copy(nodes, std::back_inserter(m_nodes));
    std::ranges::sort(m_nodes);
    m_n_owned_nodes = m_nodes.size();
}

MeshPartition::MeshPartition(domain_map_t                                 domains,
                             SizedRangeOfConvertibleTo_c< n_id_t > auto&& nodes,
                             SizedRangeOfConvertibleTo_c< n_id_t > auto&& ghost_nodes)
    : m_domains{std::move(domains)}, m_n_owned_nodes{std::ranges::size(nodes)}
{
    m_nodes.reserve(std::ranges::size(nodes) + std::ranges::size(ghost_nodes));
    std::ranges::copy(std::forward< decltype(nodes) >(nodes), std::back_inserter(m_nodes));
    std::ranges::copy(std::forward< decltype(ghost_nodes) >(ghost_nodes), std::back_inserter(m_nodes));
}

MeshPartition::cfind_result_t MeshPartition::constifyFindResult(find_result_t found)
{
    if (not found)
        return {};
    else
        return std::make_pair(*detail::constifyFound(found->first), found->second);
}

const MetisGraphWrapper& MeshPartition::initDualGraph()
{
    return m_dual_graph ? *m_dual_graph : m_dual_graph.emplace(makeMetisDualGraph());
}

const MetisGraphWrapper& MeshPartition::getDualGraph() const
{
    if (m_dual_graph)
        return *m_dual_graph;
    else
        throw std::runtime_error{"Attempting to access dual graph before initialization"};
}

// Implementation note: It should be possible to sequentially traverse elements in a deterministic order (e.g. the mesh
// partitioning facilities rely on this). However, std::execution::sequenced_policy does not make this guarantee (it
// only guarantees sequential execution). For this reason, the code below contains compile-time conditionals which
// ensure that the "classic" (i.e. deterministically sequenced) traversal algorithms is called when std::execution::seq
// is passed to the member function.
template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_elements auto&& element_visitor, d_id_t domain_id, ExecPolicy&& policy)
{
    visit(std::forward< decltype(element_visitor) >(element_visitor),
          std::views::single(domain_id),
          std::forward< ExecPolicy >(policy));
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_const_elements auto&& element_visitor,
                          d_id_t                             domain_id,
                          ExecPolicy&&                       policy) const
{
    visit(std::forward< decltype(element_visitor) >(element_visitor),
          std::views::single(domain_id),
          std::forward< ExecPolicy >(policy));
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_elements auto&&     element_visitor,
                          detail::DomainPredicate_c auto&& domain_predicate,
                          ExecPolicy&&                     policy)
{
    const auto visit_domain = [&](auto& map_entry) {
        auto& [id, dom] = map_entry;
        if (domain_predicate(DomainView{dom, id}))
            dom.visit(element_visitor, policy);
    };
    visitImpl(visit_domain, m_domains, getDomainIds(), policy);
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_const_elements auto&& element_visitor,
                          detail::DomainPredicate_c auto&&   domain_predicate,
                          ExecPolicy&&                       policy) const
{
    const auto visit_domain = [&](const auto& map_entry) {
        const auto& [id, dom] = map_entry;
        if (domain_predicate(DomainView{dom, id}))
            dom.visit(element_visitor, policy);
    };
    visitImpl(visit_domain, m_domains, getDomainIds(), policy);
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_elements_and< const DomainView > auto&& element_visitor,
                          detail::DomainPredicate_c auto&&                     domain_predicate,
                          ExecPolicy&&                                         policy)
{
    const auto visit_domain = [&](auto& map_entry) {
        auto& [id, dom]        = map_entry;
        const auto domain_view = DomainView{dom, id};
        if (domain_predicate(domain_view))
            dom.visit([&](auto& element) { element_visitor(element, domain_view); }, policy);
    };
    visitImpl(visit_domain, m_domains, getDomainIds(), policy);
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_const_elements_and< const DomainView > auto&& element_visitor,
                          detail::DomainPredicate_c auto&&                           domain_predicate,
                          ExecPolicy&&                                               policy) const
{
    const auto visit_domain = [&](const auto& map_entry) {
        const auto& [id, dom]  = map_entry;
        const auto domain_view = DomainView{dom, id};
        if (domain_predicate(domain_view))
            dom.visit([&](const auto& element) { element_visitor(element, domain_view); }, policy);
    };
    visitImpl(visit_domain, m_domains, getDomainIds(), policy);
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_elements auto&& element_visitor, ExecPolicy&& policy)
{
    const auto visit_domain = [&](auto& map_entry) {
        map_entry.second.visit(element_visitor, policy);
    };
    visitImpl(visit_domain, m_domains, getDomainIds(), policy);
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_const_elements auto&& element_visitor, ExecPolicy&& policy) const
{
    const auto visit_domain = [&](const auto& map_entry) {
        map_entry.second.visit(element_visitor, policy);
    };
    visitImpl(visit_domain, m_domains, getDomainIds(), policy);
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_elements_and< const DomainView > auto&& element_visitor, ExecPolicy&& policy)
{
    const auto visit_domain = [&](auto& map_entry) {
        auto& [id, dom]        = map_entry;
        const auto domain_view = DomainView{dom, id};
        dom.visit([&](auto& element) { element_visitor(element, domain_view); }, policy);
    };
    visitImpl(visit_domain, m_domains, getDomainIds(), policy);
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_const_elements_and< const DomainView > auto&& element_visitor,
                          ExecPolicy&&                                               policy) const
{
    const auto visit_domain = [&](const auto& map_entry) {
        const auto& [id, dom]  = map_entry;
        const auto domain_view = DomainView{dom, id};
        dom.visit([&](const auto& element) { element_visitor(element, domain_view); }, policy);
    };
    visitImpl(visit_domain, m_domains, getDomainIds(), policy);
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_elements auto&&   element_visitor,
                          detail::DomainIdRange_c auto&& domain_ids,
                          ExecPolicy&&                   policy)
{
    const auto visit_domain = [&](auto& map_entry) {
        map_entry.second.visit(element_visitor, policy);
    };
    visitImpl(visit_domain, m_domains, std::forward< decltype(domain_ids) >(domain_ids), policy);
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_const_elements auto&& element_visitor,
                          detail::DomainIdRange_c auto&&     domain_ids,
                          ExecPolicy&&                       policy) const
{
    const auto visit_domain = [&](const auto& map_entry) {
        map_entry.second.visit(element_visitor, policy);
    };
    visitImpl(visit_domain, m_domains, std::forward< decltype(domain_ids) >(domain_ids), policy);
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_elements_and< const DomainView > auto&& element_visitor,
                          detail::DomainIdRange_c auto&&                       domain_ids,
                          ExecPolicy&&                                         policy)
{
    const auto visit_domain = [&](auto& map_entry) {
        const auto domain_view = DomainView{map_entry.second, map_entry.first};
        map_entry.second.visit([&](auto& element) { element_visitor(element, domain_view); }, policy);
    };
    visitImpl(visit_domain, m_domains, std::forward< decltype(domain_ids) >(domain_ids), policy);
}

template < SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition::visit(invocable_on_const_elements_and< const DomainView > auto&& element_visitor,
                          detail::DomainIdRange_c auto&&                             domain_ids,
                          ExecPolicy&&                                               policy) const
{
    const auto visit_domain = [&](const auto& map_entry) {
        const auto domain_view = DomainView{map_entry.second, map_entry.first};
        map_entry.second.visit([&](const auto& element) { element_visitor(element, domain_view); }, policy);
    };
    visitImpl(visit_domain, m_domains, std::forward< decltype(domain_ids) >(domain_ids), policy);
}

template < SimpleExecutionPolicy_c ExecPolicy >
auto MeshPartition::transformReduce(const auto&                    zero,
                                    auto&&                         reduction,
                                    auto&&                         transform,
                                    detail::DomainIdRange_c auto&& domain_ids,
                                    ExecPolicy&&                   policy) const -> std::decay_t< decltype(zero) >
    requires invocable_on_const_elements_r< decltype(transform), std::decay_t< decltype(zero) > > and
             requires {
                 {
                     std::invoke(reduction, zero, zero)
                 } -> std::convertible_to< std::decay_t< decltype(zero) > >;
             }
{
    const auto reduce_domain = [&](d_id_t id) {
        return m_domains.at(id).transformReduce(zero, reduction, transform, policy);
    };
    if constexpr (SequencedPolicy_c< ExecPolicy >)
    {
        auto ids_to_reduce = std::forward< decltype(domain_ids) >(domain_ids) |
                             std::views::filter([&](d_id_t id) { return m_domains.contains(id); }) | std::views::common;
        return std::transform_reduce(begin(ids_to_reduce), end(ids_to_reduce), zero, reduction, reduce_domain);
    }
    else
    {
        const auto ids_to_reduce = filterExistingDomainIds(m_domains, std::forward< decltype(domain_ids) >(domain_ids));
        return util::tbb::parallelTransformReduce(
            ids_to_reduce, zero, std::forward< decltype(reduction) >(reduction), reduce_domain);
    }
}

template < SimpleExecutionPolicy_c ExecPolicy >
MeshPartition::find_result_t MeshPartition::find(invocable_on_const_elements_r< bool > auto&& predicate,
                                                 ExecPolicy&&                                 policy)
{
    return find(
        std::forward< decltype(predicate) >(predicate),
        [](const DomainView&) { return true; },
        std::forward< ExecPolicy >(policy));
}

template < SimpleExecutionPolicy_c ExecPolicy >
MeshPartition::cfind_result_t MeshPartition::find(invocable_on_const_elements_r< bool > auto&& predicate,
                                                  ExecPolicy&&                                 policy) const
{
    return constifyFindResult(const_cast< MeshPartition* >(this)->find(std::forward< decltype(predicate) >(predicate),
                                                                       std::forward< ExecPolicy >(policy)));
}

template < SimpleExecutionPolicy_c ExecPolicy >
MeshPartition::find_result_t MeshPartition::find(invocable_on_const_elements_r< bool > auto&& predicate,
                                                 detail::DomainPredicate_c auto&&             domain_predicate,
                                                 ExecPolicy&&                                 policy)
{
    find_result_t result;
    std::mutex    mut;
    (void)std::find_if(policy, begin(m_domains), end(m_domains), [&](auto& map_entry) {
        auto& [id, domain]     = map_entry;
        const auto find_result = domain.find(predicate, policy);
        if (find_result)
        {
            std::lock_guard lock{mut};
            result = std::make_pair(*find_result, id);
        }
        return find_result.has_value();
    });
    return result;
}

template < SimpleExecutionPolicy_c ExecPolicy >
MeshPartition::cfind_result_t MeshPartition::find(invocable_on_const_elements_r< bool > auto&& predicate,
                                                  detail::DomainPredicate_c auto&&             domain_ids,
                                                  ExecPolicy&&                                 policy) const
{
    return constifyFindResult(const_cast< MeshPartition* >(this)->find(std::forward< decltype(predicate) >(predicate),
                                                                       std::forward< decltype(domain_ids) >(domain_ids),
                                                                       std::forward< ExecPolicy >(policy)));
}

MeshPartition::find_result_t MeshPartition::find(el_id_t id)
{
    for (auto& [dom_id, dom] : m_domains)
    {
        const auto find_result = dom.find(id);
        if (find_result)
            return std::make_pair(*find_result, dom_id);
    }
    return {};
}

MeshPartition::cfind_result_t MeshPartition::find(el_id_t id) const
{
    return constifyFindResult(const_cast< MeshPartition* >(this)->find(id));
}

template < ElementTypes T, el_o_t O >
MeshPartition::el_boundary_view_result_t MeshPartition::getElementBoundaryViewImpl(const Element< T, O >& el) const
{
    constexpr auto miss       = std::numeric_limits< el_side_t >::max();
    constexpr auto matchElDim = []< ElementTypes T_, el_o_t O_ >(const Element< T_, O_ >*) {
        return Element< T_, O_ >::native_dim - 1 == Element< T, O >::native_dim;
    };

    const auto boundary_nodes = getSortedArray(el.getNodes());
    const auto match_side     = [&]< ElementTypes T_, el_o_t O_ >(const Element< T_, O_ >* domain_element) {
        return detail::matchBoundaryNodesToElement(*domain_element, boundary_nodes);
    };

    const auto adjacent_elems = m_dual_graph->getElementAdjacent(el.getId());
    for (el_id_t adj_id : adjacent_elems)
    {
        const auto  find_result = find(adj_id);
        const auto& adj_el      = find_result->first;
        if (not std::visit(matchElDim, adj_el))
            continue;
        const auto side_index = std::visit(match_side, adj_el);
        if (side_index != miss)
            return std::make_pair(find_result, side_index);
    }
    return {{}, miss};
}

template < ElementTypes T, el_o_t O >
MeshPartition::el_boundary_view_result_t MeshPartition::getElementBoundaryViewFallback(const Element< T, O >& el,
                                                                                       d_id_t                 d) const
{
    const auto boundary_nodes = getSortedArray(el.getNodes());
    el_side_t  side_index     = 0;

    const auto is_domain_element = [&]< ElementTypes T_, el_o_t O_ >(const Element< T_, O_ >& domain_element) {
        side_index = detail::matchBoundaryNodesToElement(domain_element, boundary_nodes);
        return side_index != std::numeric_limits< el_side_t >::max();
    };

    return std::make_pair(
        find(is_domain_element, [&](const DomainView& dv) { return dv.getDim() == getDomainView(d).getDim() + 1; }),
        side_index);
}

template < ElementTypes T, el_o_t O >
MeshPartition::el_boundary_view_result_t MeshPartition::getElementBoundaryView(const Element< T, O >& el,
                                                                               d_id_t                 d) const
{
    if (isDualGraphInitialized())
        return getElementBoundaryViewImpl(el);
    else
        return getElementBoundaryViewFallback(el, d);
}

BoundaryView MeshPartition::getBoundaryView(detail::DomainIdRange_c auto&& boundary_ids) const
{
    size_t n_boundary_els = 0;
    for (d_id_t id : boundary_ids)
    {
        const auto dom_it = m_domains.find(id);
        if (dom_it != end(m_domains))
            n_boundary_els += dom_it->second.getNElements();
    }
    BoundaryView::boundary_element_view_variant_vector_t boundary_elements;
    boundary_elements.reserve(n_boundary_els);
    std::mutex       insertion_mutex;
    std::atomic_bool error_flag{false};

    for (d_id_t id : boundary_ids)
    {
        const auto insert_boundary_element_view = [&](const auto& boundary_element) {
            const auto& [domain_element_variant_opt, side_index] = getElementBoundaryView(boundary_element, id);
            if (not domain_element_variant_opt)
            {
                error_flag.store(true);
                return;
            }

            const auto emplace_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >* domain_element_ptr) {
                std::scoped_lock lock{insertion_mutex};
                boundary_elements.emplace_back(
                    std::in_place_type< BoundaryElementView< T, O > >, *domain_element_ptr, side_index);
            };
            std::visit(emplace_element, domain_element_variant_opt->first);
        };
        visit(insert_boundary_element_view, id, std::execution::par);
        if (error_flag.load())
            throw std::runtime_error{
                "BoundaryView could not be constructed because some of the boundary elements are not edges/faces of "
                "any of the domain elements in the partition. This may be because the mesh was partitioned with "
                "incorrectly specified boundaries, resulting in the edge/face element being in a different partition "
                "than its parent area/volume element."};
    }
    return BoundaryView{std::move(boundary_elements), *this};
}

size_t MeshPartition::getNElements() const
{
    return std::accumulate(
        m_domains.cbegin(), m_domains.cend(), 0, [](size_t s, const auto& d) { return s + d.second.getNElements(); });
}

template < el_o_t O >
MeshPartition::domain_map_t MeshPartition::getConversionAlloc() const
{
    domain_map_t retval;
    for (const auto& [id, dom] : m_domains)
        retval.emplace_hint(retval.cend(), id, dom.template getConversionAlloc< O >());
    return retval;
}

auto MeshPartition::convertToMetisFormat() const
{
    std::vector< idx_t > eind, eptr;
    size_t               topo_size = 0;
    visit([&](const auto& element) { topo_size += element.getNodes().size(); });
    eind.reserve(topo_size);
    eptr.reserve(getNElements() + 1);
    eptr.push_back(0);
    for (el_id_t id = 0; id < getNElements(); ++id)
    {
        const auto el_ptr = find(id)->first;
        std::visit(
            [&]< ElementTypes T, el_o_t O >(const Element< T, O >* element) {
                std::ranges::copy(element->getNodes() |
                                      std::views::transform([](auto n) { return exactIntegerCast< idx_t >(n); }),
                                  std::back_inserter(eind));
                eptr.push_back(exactIntegerCast< idx_t >(eptr.back() + element->getNodes().size()));
            },
            el_ptr);
    };
    return std::make_pair(std::move(eptr), std::move(eind));
}

MetisGraphWrapper MeshPartition::makeMetisDualGraph() const
{
    auto [eptr, eind] = convertToMetisFormat();
    auto  ne          = exactIntegerCast< idx_t >(getNElements());
    auto  nn          = exactIntegerCast< idx_t >(getOwnedNodes().size());
    idx_t ncommon     = 2;
    idx_t numflag     = 0;

    idx_t* xadj{};
    idx_t* adjncy{};

    const auto error_code = METIS_MeshToDual(&ne, &nn, eptr.data(), eind.data(), &ncommon, &numflag, &xadj, &adjncy);
    detail::handleMetisErrorCode(error_code);

    return MetisGraphWrapper{xadj, adjncy, getNElements()};
}

size_t MeshPartition::computeTopoHash() const
{
    constexpr auto hash_range = [](std::ranges::contiguous_range auto&& r) -> size_t {
        const auto data = std::span{std::forward< decltype(r) >(r)};
        return robin_hood::hash_bytes(data.data(), data.size_bytes());
    };
    const auto hash_element = [&](const auto& element) {
        return hash_range(element.getNodes());
    };
    const size_t topo_hash =
        transformReduce(size_t{}, std::bit_xor<>{}, hash_element, getDomainIds(), std::execution::par);
    return topo_hash ^ hash_range(m_nodes);
}

void MeshPartition::visitImpl(auto&&                         visitor,
                              auto&&                         domain_map,
                              auto&&                         domain_ids,
                              SimpleExecutionPolicy_c auto&& policy)
{
    if constexpr (SequencedPolicy_c< decltype(policy) >)
        std::ranges::for_each(std::forward< decltype(domain_ids) >(domain_ids) |
                                  std::views::transform([&](d_id_t id) { return domain_map.find(id); }) |
                                  std::views::filter([&](auto map_iter) { return map_iter != domain_map.end(); }) |
                                  std::views::transform([&](auto map_iter) -> decltype(auto) { return *map_iter; }),
                              std::forward< decltype(visitor) >(visitor));
    else
    {
        const auto ids_to_visit = filterExistingDomainIds(domain_map, std::forward< decltype(domain_ids) >(domain_ids));
        switch (ids_to_visit.size())
        {
        case 0:
            break;
        case 1: // Small optimization: bypass the TBB scheduler, invoke visitor directly
            std::invoke(std::forward< decltype(visitor) >(visitor), *domain_map.find(ids_to_visit.front()));
            break;
        default:
            util::tbb::parallelFor(ids_to_visit, [&](d_id_t id) { std::invoke(visitor, *domain_map.find(id)); });
        }
    }
}

auto MeshPartition::filterExistingDomainIds(const auto& domain_map, detail::DomainIdRange_c auto&& ids)
    -> std::vector< d_id_t >
{
    std::vector< d_id_t > retval;
    retval.reserve(std::ranges::size(ids));
    std::ranges::copy(std::forward< decltype(ids) >(ids) |
                          std::views::filter([&](d_id_t id) { return domain_map.contains(id); }),
                      std::back_inserter(retval));
    return retval;
}
} // namespace lstr
#endif // L3STER_MESH_MESHPARTITION_HPP
