#ifndef L3STER_MESH_MESHPARTITION_HPP
#define L3STER_MESH_MESHPARTITION_HPP

#include "l3ster/mesh/Domain.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/MetisUtils.hpp"
#include "l3ster/util/Ranges.hpp"

#include "l3ster/util/RobinHoodHashTables.hpp"

#include <map>

namespace lstr::mesh
{
template < typename T, el_o_t... orders >
concept DomainPredicate_c = requires(T op, DomainView< orders... > dv) {
    {
        std::invoke(op, dv)
    } -> std::convertible_to< bool >;
};

template < el_o_t... orders >
class MeshPartition
{
    static_assert(sizeof...(orders) > 0);
    using DefaultExec = const std::execution::sequenced_policy&;

public:
    using domain_map_t        = std::map< d_id_t, Domain< orders... > >;
    using find_result_t       = Domain< orders... >::find_result_t;
    using const_find_result_t = Domain< orders... >::const_find_result_t;
    using node_span_t         = std::span< const n_id_t >;

    friend struct SerializedPartition;

    MeshPartition() = default;
    inline MeshPartition(domain_map_t domains);
    MeshPartition(domain_map_t domains, std::vector< n_id_t > nodes, size_t n_owned_nodes)
        : m_domains{std::move(domains)}, m_nodes{std::move(nodes)}, m_n_owned_nodes{n_owned_nodes}
    {}
    template < SizedRangeOfConvertibleTo_c< n_id_t > Owned, SizedRangeOfConvertibleTo_c< n_id_t > Ghost >
    MeshPartition(domain_map_t domains, Owned&& nodes, Ghost&& ghost_nodes);

    // Iteration (visiting) over elements
    template < MutableElementVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    void visit(Visitor&& element_visitor, ExecPolicy&& policy = {});
    template < ConstElementVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    void visit(Visitor&& element_visitor, ExecPolicy&& policy = {}) const;
    template < MutableElementVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    void visit(Visitor&& element_visitor, d_id_t domain_id, ExecPolicy&& policy = {});
    template < ConstElementVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    void visit(Visitor&& element_visitor, d_id_t domain_id, ExecPolicy&& policy = {}) const;
    template < MutableElementVisitor_c< orders... > F,
               DomainIdRange_c                      Ids,
               SimpleExecutionPolicy_c              ExecPolicy = DefaultExec >
    void visit(F&& element_visitor, Ids&& domain_ids, ExecPolicy&& policy = {});
    template < ConstElementVisitor_c< orders... > F,
               DomainIdRange_c                    Ids,
               SimpleExecutionPolicy_c            ExecPolicy = DefaultExec >
    void visit(F&& element_visitor, Ids&& domain_ids, ExecPolicy&& policy = {}) const;

    // Reduction over elements
    // Note: `zero` must be the identity element for the reduction (as opposed to, e.g., std::transform_reduce)
    // Note: The iteration order is indeterminate, even if std::execution::seq is passed
    template < DomainIdRange_c         IdRange,
               std::copy_constructible Zero,
               std::copy_constructible Transform,
               std::copy_constructible Reduction  = std::plus<>,
               SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    auto transformReduce(
        IdRange&& domain_ids, Zero zero, Transform transform, Reduction reduction = {}, ExecPolicy&& policy = {}) const
        -> Zero
        requires TransformReducible_c< Zero, Transform, Reduction, orders... >;

    // find
    // Note: if the predicate returns true for multiple elements, it is undefined which one is returned
    template < ElementPredicate_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    auto find(F&& predicate, ExecPolicy&& policy = {}) -> find_result_t;
    template < ElementPredicate_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy = DefaultExec >
    auto find(F&& predicate, ExecPolicy&& policy = {}) const -> const_find_result_t;
    template < ElementPredicate_c< orders... > F,
               DomainIdRange_c                 Ids,
               SimpleExecutionPolicy_c         ExecPolicy = DefaultExec >
    auto find(F&& predicate, Ids&& ids, ExecPolicy&& policy = {}) -> find_result_t;
    template < ElementPredicate_c< orders... > F,
               DomainIdRange_c                 Ids,
               SimpleExecutionPolicy_c         ExecPolicy = DefaultExec >
    auto                       find(F&& predicate, Ids&& ids, ExecPolicy&& policy = {}) const -> const_find_result_t;
    inline find_result_t       find(el_id_t id);
    inline const_find_result_t find(el_id_t id) const;

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
    template < DomainIdRange_c Ids >
    static auto filterExistingDomainIds(const domain_map_t& domain_map, Ids&& ids) -> std::vector< d_id_t >;
    // Deduce constness based on the domain map, helps with deduplication. Idea similar to C++23 "deducing this"
    static void visitImpl(auto&& visitor, auto&& domain_map, auto&& domain_ids, auto&& policy);

    domain_map_t          m_domains;
    std::vector< n_id_t > m_nodes;
    size_t                m_n_owned_nodes;
};

template < el_o_t... orders >
MeshPartition< orders... >::MeshPartition(MeshPartition< orders... >::domain_map_t domains_)
    : m_domains{std::move(domains_)}
{
    robin_hood::unordered_flat_set< n_id_t > nodes;
    visit([&](const auto& element) { std::ranges::for_each(element.getNodes(), [&](n_id_t n) { nodes.insert(n); }); });
    m_nodes.reserve(nodes.size());
    std::ranges::copy(nodes, std::back_inserter(m_nodes));
    std::ranges::sort(m_nodes);
    m_n_owned_nodes = m_nodes.size();
}

template < el_o_t... orders >
template < SizedRangeOfConvertibleTo_c< n_id_t > Owned, SizedRangeOfConvertibleTo_c< n_id_t > Ghost >
MeshPartition< orders... >::MeshPartition(domain_map_t domains, Owned&& owned_nodes, Ghost&& ghost_nodes)
    : m_domains{std::move(domains)}, m_n_owned_nodes{std::ranges::size(owned_nodes)}
{
    m_nodes.reserve(std::ranges::size(owned_nodes) + std::ranges::size(ghost_nodes));
    auto back_inserter = std::back_inserter(m_nodes);
    std::ranges::copy(std::forward< Owned >(owned_nodes), back_inserter);
    std::ranges::copy(std::forward< Ghost >(ghost_nodes), back_inserter);
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
template < MutableElementVisitor_c< orders... > F, DomainIdRange_c Ids, SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition< orders... >::visit(F&& element_visitor, Ids&& domain_ids, ExecPolicy&& policy)
{
    visitImpl(std::forward< F >(element_visitor),
              m_domains,
              std::forward< Ids >(domain_ids),
              std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < ConstElementVisitor_c< orders... > F, DomainIdRange_c Ids, SimpleExecutionPolicy_c ExecPolicy >
void MeshPartition< orders... >::visit(F&& element_visitor, Ids&& domain_ids, ExecPolicy&& policy) const
{
    visitImpl(std::forward< F >(element_visitor),
              m_domains,
              std::forward< Ids >(domain_ids),
              std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < DomainIdRange_c         IdRange,
           std::copy_constructible Zero,
           std::copy_constructible Transform,
           std::copy_constructible Reduction,
           SimpleExecutionPolicy_c ExecPolicy >
auto MeshPartition< orders... >::transformReduce(
    IdRange&& domain_ids, Zero zero, Transform transform, Reduction reduction, ExecPolicy&& policy) const -> Zero
    requires TransformReducible_c< Zero, Transform, Reduction, orders... >
{
    const auto ids           = filterExistingDomainIds(m_domains, std::forward< IdRange >(domain_ids));
    const auto reduce_domain = [&](d_id_t id) {
        return m_domains.at(id).transformReduce(zero, transform, reduction, policy);
    };
    return std::transform_reduce(policy, ids.begin(), ids.end(), std::move(zero), std::move(reduction), reduce_domain);
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy >
auto MeshPartition< orders... >::find(F&& predicate, ExecPolicy&& policy) -> find_result_t
{
    const auto dom_ids = util::toVector(getDomainIds());
    return find(std::forward< decltype(predicate) >(predicate), dom_ids, std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > F, SimpleExecutionPolicy_c ExecPolicy >
auto MeshPartition< orders... >::find(F&& predicate, ExecPolicy&& policy) const -> const_find_result_t
{
    const auto dom_ids = util::toVector(getDomainIds());
    return find(std::forward< decltype(predicate) >(predicate), dom_ids, std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > F, DomainIdRange_c Ids, SimpleExecutionPolicy_c ExecPolicy >
auto MeshPartition< orders... >::find(F&& predicate, Ids&& ids, ExecPolicy&& policy) -> find_result_t
{
    const auto const_find = std::as_const(*this).find(
        std::forward< F >(predicate), std::forward< Ids >(ids), std::forward< ExecPolicy >(policy));
    return detail::makeMutableFindResult< orders... >(const_find);
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > F, DomainIdRange_c Ids, SimpleExecutionPolicy_c ExecPolicy >
auto MeshPartition< orders... >::find(F&& predicate, Ids&& ids, ExecPolicy&& policy) const -> const_find_result_t
{
    auto ids_common = std::forward< Ids >(ids) | std::views::common;
    auto retval     = const_find_result_t{};
    auto mut        = std::mutex{};
    std::find_if(policy, ids_common.begin(), ids_common.end(), [&](d_id_t id) {
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
    constexpr auto hash_range = [](std::ranges::contiguous_range auto&& r) -> size_t {
        const auto data = std::span{std::forward< decltype(r) >(r)};
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
void MeshPartition< orders... >::visitImpl(auto&& visitor, auto&& domain_map, auto&& domain_ids, auto&& policy)
{
    const auto ids_to_visit    = filterExistingDomainIds(domain_map, std::forward< decltype(domain_ids) >(domain_ids));
    const auto visit_domain_id = [&](d_id_t id) {
        domain_map.at(id).visit(visitor, policy);
    };
    std::for_each(std::forward< decltype(policy) >(policy), ids_to_visit.begin(), ids_to_visit.end(), visit_domain_id);
}

template < el_o_t... orders >
template < DomainIdRange_c Ids >
auto MeshPartition< orders... >::filterExistingDomainIds(const domain_map_t& domain_map, Ids&& ids)
    -> std::vector< d_id_t >
{
    return util::toVector(util::castView< d_id_t >(
        std::forward< Ids >(ids) | std::views::filter([&](d_id_t id) { return domain_map.contains(id); })));
}
} // namespace lstr::mesh
#endif // L3STER_MESH_MESHPARTITION_HPP
