#ifndef L3STER_MESH_MESHPARTITION_HPP
#define L3STER_MESH_MESHPARTITION_HPP

#include "l3ster/mesh/Domain.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/ArrayOwner.hpp"
#include "l3ster/util/MetisUtils.hpp"
#include "l3ster/util/Ranges.hpp"

#include "l3ster/util/RobinHoodHashTables.hpp"

#include <map>

namespace lstr::mesh
{
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
    template < el_o_t... O >
    friend auto copy(const MeshPartition< O... >&) -> MeshPartition< O... >;

    MeshPartition() = default;
    inline MeshPartition(domain_map_t domains);
    MeshPartition(domain_map_t domains, util::ArrayOwner< n_id_t > nodes, size_t n_owned_nodes)
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
    // Deduce constness based on the domain map, helps with deduplication. Idea similar to C++23 "deducing this"
    template < typename Visitor, typename DomainMap, typename Policy >
    static void
    visitImpl(Visitor&& visitor, DomainMap&& domain_map, const util::ArrayOwner< d_id_t >& domain_ids, Policy&& policy);

    domain_map_t               m_domains;
    util::ArrayOwner< n_id_t > m_nodes;
    size_t                     m_n_owned_nodes;
};

template < el_o_t... orders >
auto copy(const MeshPartition< orders... >& mesh) -> MeshPartition< orders... >
{
    return {mesh.m_domains, copy(mesh.m_nodes), mesh.m_n_owned_nodes};
}

template < el_o_t... orders >
MeshPartition< orders... >::MeshPartition(MeshPartition< orders... >::domain_map_t domains)
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
}

template < el_o_t... orders >
template < SizedRangeOfConvertibleTo_c< n_id_t > Owned, SizedRangeOfConvertibleTo_c< n_id_t > Ghost >
MeshPartition< orders... >::MeshPartition(domain_map_t domains, Owned&& owned_nodes, Ghost&& ghost_nodes)
    : m_domains{std::move(domains)},
      m_nodes(std::ranges::size(owned_nodes) + std::ranges::size(ghost_nodes)),
      m_n_owned_nodes{std::ranges::size(owned_nodes)}
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
} // namespace lstr::mesh
#endif // L3STER_MESH_MESHPARTITION_HPP
