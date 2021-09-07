#ifndef L3STER_MESH_MESHPARTITION_HPP
#define L3STER_MESH_MESHPARTITION_HPP

#include "mesh/BoundaryView.hpp"
#include "mesh/Domain.hpp"
#include "mesh/ElementSideMatching.hpp"
#include "util/Algorithm.hpp"
#include "util/MetisUtils.hpp"

#include <map>
#include <variant>

namespace lstr
{
namespace detail
{
template < typename T >
concept domain_predicate = requires(T op, const DomainView dv)
{
    {
        op(dv)
        } -> std::convertible_to< bool >;
};

template < typename T >
concept inv_cel_dv = invocable_on_const_elements_and< T, const DomainView >;
} // namespace detail

class MeshPartition
{
public:
    using domain_map_t              = std::map< d_id_t, Domain >;
    using node_vec_t                = std::vector< n_id_t >;
    using find_result_t             = std::optional< std::pair< element_ptr_variant_t, d_id_t > >;
    using cfind_result_t            = std::optional< std::pair< element_cptr_variant_t, d_id_t > >;
    using el_boundary_view_result_t = std::pair< cfind_result_t, el_ns_t >;

    MeshPartition(const MeshPartition&)     = delete;
    MeshPartition(MeshPartition&&) noexcept = default;
    MeshPartition& operator=(const MeshPartition&) = delete;
    MeshPartition& operator=(MeshPartition&&) noexcept = default;
    ~MeshPartition()                                   = default;

    inline explicit MeshPartition(domain_map_t domains_);
    MeshPartition(domain_map_t domains_, node_vec_t nodes_, node_vec_t ghost_nodes_)
        : domains{std::move(domains_)}, nodes{std::move(nodes_)}, ghost_nodes{std::move(ghost_nodes_)}
    {}

    inline const MetisGraphWrapper& initDualGraph();
    void                            deleteDualGraph() noexcept { dual_graph.reset(); }
    [[nodiscard]] bool              isDualGraphInitialized() const noexcept { return dual_graph.has_value(); }
    [[nodiscard]] inline const MetisGraphWrapper& getDualGraph() const;

    template < invocable_on_elements    F,
               detail::domain_predicate D,
               ExecutionPolicy_c        ExecPolicy = std::execution::sequenced_policy >
    decltype(auto) visit(F&& element_visitor, D&& domain_predicate, const ExecPolicy& policy = std::execution::seq);
    template < invocable_on_const_elements F,
               detail::domain_predicate    D,
               ExecutionPolicy_c           ExecPolicy = std::execution::sequenced_policy >
    decltype(auto)
    cvisit(F&& element_visitor, D&& domain_predicate, const ExecPolicy& policy = std::execution::seq) const;
    template < invocable_on_elements_and< const DomainView > F,
               detail::domain_predicate                      D,
               ExecutionPolicy_c                             ExecPolicy = std::execution::sequenced_policy >
    decltype(auto) visit(F&& element_visitor, D&& domain_predicate, const ExecPolicy& policy = std::execution::seq);
    template < detail::inv_cel_dv       F,
               detail::domain_predicate D,
               ExecutionPolicy_c        ExecPolicy = std::execution::sequenced_policy >
    decltype(auto)
    cvisit(F&& element_visitor, D&& domain_predicate, const ExecPolicy& policy = std::execution::seq) const;
    template < invocable_on_elements F, ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    decltype(auto) visit(F&& element_visitor, const ExecPolicy& policy = std::execution::seq);
    template < invocable_on_const_elements F, ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    decltype(auto) cvisit(F&& element_visitor, const ExecPolicy& policy = std::execution::seq) const;
    template < invocable_on_elements_and< const DomainView > F,
               ExecutionPolicy_c                             ExecPolicy = std::execution::sequenced_policy >
    decltype(auto) visit(F&& element_visitor, const ExecPolicy& policy = std::execution::seq);
    template < detail::inv_cel_dv F, ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    decltype(auto) cvisit(F&& element_visitor, const ExecPolicy& policy = std::execution::seq) const;
    template < invocable_on_elements F, ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    decltype(auto)
    visit(F&& element_visitor, const std::vector< d_id_t >& domain_ids, const ExecPolicy& policy = std::execution::seq);
    template < invocable_on_const_elements F, ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    decltype(auto) cvisit(F&&                          element_visitor,
                          const std::vector< d_id_t >& domain_ids,
                          const ExecPolicy&            policy = std::execution::seq) const;
    template < invocable_on_elements_and< const DomainView > F,
               ExecutionPolicy_c                             ExecPolicy = std::execution::sequenced_policy >
    decltype(auto)
    visit(F&& element_visitor, const std::vector< d_id_t >& domain_ids, const ExecPolicy& policy = std::execution::seq);
    template < detail::inv_cel_dv F, ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    decltype(auto) cvisit(F&&                          element_visitor,
                          const std::vector< d_id_t >& domain_ids,
                          const ExecPolicy&            policy = std::execution::seq) const;

    // Note: if the predicate returns true for multiple elements, the reference to any one of them may be returned
    template < invocable_on_const_elements_r< bool > F,
               ExecutionPolicy_c                     ExecPolicy = std::execution::sequenced_policy >
    [[nodiscard]] find_result_t find(F&& predicate, const ExecPolicy& policy = std::execution::seq);
    template < invocable_on_const_elements_r< bool > F,
               ExecutionPolicy_c                     ExecPolicy = std::execution::sequenced_policy >
    [[nodiscard]] cfind_result_t find(F&& predicate, const ExecPolicy& policy = std::execution::seq) const;
    template < invocable_on_const_elements_r< bool > F,
               ExecutionPolicy_c                     ExecPolicy = std::execution::sequenced_policy >
    [[nodiscard]] find_result_t
    find(F&& predicate, const std::vector< d_id_t >& domain_ids, const ExecPolicy& policy = std::execution::seq);
    template < invocable_on_const_elements_r< bool > F,
               ExecutionPolicy_c                     ExecPolicy = std::execution::sequenced_policy >
    [[nodiscard]] cfind_result_t
    find(F&& predicate, const std::vector< d_id_t >& domain_ids, const ExecPolicy& policy = std::execution::seq) const;
    template < invocable_on_const_elements_r< bool > F,
               typename D,
               ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    [[nodiscard]] find_result_t
    find(F&& predicate, D&& domain_predicate, const ExecPolicy& policy = std::execution::seq);
    template < invocable_on_const_elements_r< bool > F,
               typename D,
               ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    [[nodiscard]] cfind_result_t
    find(F&& predicate, D&& domain_predicate, const ExecPolicy& policy = std::execution::seq) const;
    [[nodiscard]] inline find_result_t  find(el_id_t id);
    [[nodiscard]] inline cfind_result_t find(el_id_t id) const;

    template < ElementTypes T, el_o_t O >
    el_boundary_view_result_t         getElementBoundaryView(const Element< T, O >& el, d_id_t d) const;
    [[nodiscard]] inline BoundaryView getBoundaryView(d_id_t) const;

    [[nodiscard]] DomainView                   getDomainView(d_id_t id) const { return DomainView{domains.at(id), id}; }
    [[nodiscard]] inline size_t                getNElements() const;
    [[nodiscard]] auto                         getNDomains() const { return domains.size(); }
    [[nodiscard]] inline std::vector< d_id_t > getDomainIds() const;
    [[nodiscard]] const Domain&                getDomain(d_id_t id) const { return domains.at(id); }

    [[nodiscard]] const node_vec_t& getNodes() const noexcept { return nodes; }
    [[nodiscard]] const node_vec_t& getGhostNodes() const noexcept { return ghost_nodes; }

    template < el_o_t O >
    [[nodiscard]] domain_map_t getConversionAlloc() const;

private:
    [[nodiscard]] inline auto              convertToMetisFormat() const;
    [[nodiscard]] inline MetisGraphWrapper makeMetisDualGraph() const;
    static inline cfind_result_t           constifyFindResult(find_result_t found);

    template < ElementTypes T, el_o_t O >
    el_boundary_view_result_t getElementBoundaryViewImpl(const Element< T, O >& el) const;
    template < ElementTypes T, el_o_t O >
    el_boundary_view_result_t getElementBoundaryViewFallback(const Element< T, O >& el, d_id_t d) const;

    domain_map_t                       domains;
    node_vec_t                         nodes;
    node_vec_t                         ghost_nodes;
    std::optional< MetisGraphWrapper > dual_graph;
};

MeshPartition::cfind_result_t MeshPartition::constifyFindResult(find_result_t found)
{
    if (not found)
        return {};
    else
        return std::make_pair(*detail::constifyFound(found->first), found->second);
}

const MetisGraphWrapper& MeshPartition::initDualGraph()
{
    return dual_graph ? *dual_graph : dual_graph.emplace(makeMetisDualGraph());
}

const MetisGraphWrapper& MeshPartition::getDualGraph() const
{
    if (dual_graph)
        return *dual_graph;
    else
        throw std::runtime_error{"Attempting to access dual graph before initialization"};
}

template < invocable_on_elements F, detail::domain_predicate D, ExecutionPolicy_c ExecPolicy >
decltype(auto) MeshPartition::visit(F&& element_visitor, D&& domain_predicate, const ExecPolicy& policy)
{
    for (auto& [id, dom] : domains)
        if (domain_predicate(DomainView{dom, id}))
            dom.visit(element_visitor, policy);
    return std::forward< F >(element_visitor);
}

template < invocable_on_const_elements F, detail::domain_predicate D, ExecutionPolicy_c ExecPolicy >
decltype(auto) MeshPartition::cvisit(F&& element_visitor, D&& domain_predicate, const ExecPolicy& policy) const
{
    for (const auto& [id, dom] : domains)
        if (domain_predicate(DomainView{dom, id}))
            dom.cvisit(element_visitor, policy);
    return std::forward< F >(element_visitor);
}

template < invocable_on_elements_and< const DomainView > F, detail::domain_predicate D, ExecutionPolicy_c ExecPolicy >
decltype(auto) MeshPartition::visit(F&& element_visitor, D&& domain_predicate, const ExecPolicy& policy)
{
    Domain     dummy;
    DomainView current_view{dummy, 0};
    visit([&](auto& element) { element_visitor(element, current_view); },
          [&](const DomainView& d) {
              current_view = d;
              return domain_predicate(d);
          },
          policy);
    return std::forward< F >(element_visitor);
}

template < detail::inv_cel_dv F, detail::domain_predicate D, ExecutionPolicy_c ExecPolicy >
decltype(auto) MeshPartition::cvisit(F&& element_visitor, D&& domain_predicate, const ExecPolicy& policy) const
{
    Domain     dummy;
    DomainView current_view{dummy, 0};
    cvisit([&](const auto& element) { element_visitor(element, current_view); },
           [&](const DomainView& d) {
               current_view = d;
               return domain_predicate(d);
           },
           policy);
    return std::forward< F >(element_visitor);
}

template < invocable_on_elements F, ExecutionPolicy_c ExecPolicy >
decltype(auto) MeshPartition::visit(F&& element_visitor, const ExecPolicy& policy)
{
    return visit(
        std::forward< F >(element_visitor), [](const DomainView&) { return true; }, policy);
}

template < invocable_on_const_elements F, ExecutionPolicy_c ExecPolicy >
decltype(auto) MeshPartition::cvisit(F&& element_visitor, const ExecPolicy& policy) const
{
    return cvisit(
        std::forward< F >(element_visitor), [](const DomainView&) { return true; }, policy);
}

template < invocable_on_elements_and< const DomainView > F, ExecutionPolicy_c ExecPolicy >
decltype(auto) MeshPartition::visit(F&& element_visitor, const ExecPolicy& policy)
{
    return visit(
        std::forward< F >(element_visitor), [](const DomainView&) { return true; }, policy);
}

template < detail::inv_cel_dv F, ExecutionPolicy_c ExecPolicy >
decltype(auto) MeshPartition::cvisit(F&& element_visitor, const ExecPolicy& policy) const
{
    return cvisit(
        std::forward< F >(element_visitor), [](const DomainView&) { return true; }, policy);
}

template < invocable_on_elements F, ExecutionPolicy_c ExecPolicy >
decltype(auto)
MeshPartition::visit(F&& element_visitor, const std::vector< d_id_t >& domain_ids, const ExecPolicy& policy)
{
    const auto domain_predicate = [&domain_ids](const DomainView& d) {
        return std::any_of(cbegin(domain_ids), cend(domain_ids), [&d](const d_id_t& d1) { return d.getID() == d1; });
    };
    return visit(std::forward< F >(element_visitor), domain_predicate, policy);
}

template < invocable_on_const_elements F, ExecutionPolicy_c ExecPolicy >
decltype(auto)
MeshPartition::cvisit(F&& element_visitor, const std::vector< d_id_t >& domain_ids, const ExecPolicy& policy) const
{
    const auto domain_predicate = [&](const DomainView& d) {
        return std::any_of(cbegin(domain_ids), cend(domain_ids), [&](const d_id_t& d1) { return d.getID() == d1; });
    };
    return cvisit(std::forward< F >(element_visitor), domain_predicate, policy);
}

template < invocable_on_elements_and< const DomainView > F, ExecutionPolicy_c ExecPolicy >
decltype(auto)
MeshPartition::visit(F&& element_visitor, const std::vector< d_id_t >& domain_ids, const ExecPolicy& policy)
{
    const auto domain_predicate = [&domain_ids](const DomainView& d) {
        return std::any_of(cbegin(domain_ids), cend(domain_ids), [&d](const d_id_t& d1) { return d.getID() == d1; });
    };
    return visit(std::forward< F >(element_visitor), domain_predicate, policy);
}

template < detail::inv_cel_dv F, ExecutionPolicy_c ExecPolicy >
decltype(auto)
MeshPartition::cvisit(F&& element_visitor, const std::vector< d_id_t >& domain_ids, const ExecPolicy& policy) const
{
    const auto domain_predicate = [&](const DomainView& d) {
        return std::any_of(cbegin(domain_ids), cend(domain_ids), [&](const d_id_t& d1) { return d.getID() == d1; });
    };
    return cvisit(std::forward< F >(element_visitor), domain_predicate, policy);
}

template < invocable_on_const_elements_r< bool > F, ExecutionPolicy_c ExecPolicy >
MeshPartition::find_result_t MeshPartition::find(F&& predicate, const ExecPolicy& policy)
{
    return find(
        std::forward< F >(predicate), [](const DomainView&) { return true; }, policy);
}

template < invocable_on_const_elements_r< bool > F, ExecutionPolicy_c ExecPolicy >
MeshPartition::cfind_result_t MeshPartition::find(F&& predicate, const ExecPolicy& policy) const
{
    return constifyFindResult(const_cast< MeshPartition* >(this)->find(std::forward< F >(predicate), policy));
}

template < invocable_on_const_elements_r< bool > F, ExecutionPolicy_c ExecPolicy >
MeshPartition::find_result_t
MeshPartition::find(F&& predicate, const std::vector< d_id_t >& domain_ids, const ExecPolicy& policy)
{
    return find(
        std::forward< F >(predicate),
        [&](const auto& domain_view) {
            return std::ranges::any_of(domain_ids, [&](d_id_t d) { return d == domain_view.getID(); });
        },
        policy);
}

template < invocable_on_const_elements_r< bool > F, ExecutionPolicy_c ExecPolicy >
MeshPartition::cfind_result_t
MeshPartition::find(F&& predicate, const std::vector< d_id_t >& domain_ids, const ExecPolicy& policy) const
{
    return constifyFindResult(
        const_cast< MeshPartition* >(this)->find(std::forward< F >(predicate), domain_ids, policy));
}

template < invocable_on_const_elements_r< bool > F, typename D, ExecutionPolicy_c ExecPolicy >
MeshPartition::find_result_t MeshPartition::find(F&& predicate, D&& domain_predicate, const ExecPolicy& policy)
{
    for (auto& [dom_id, dom] : domains)
    {
        if (domain_predicate(DomainView{dom, dom_id}))
        {
            const auto find_result = dom.find(predicate, policy);
            if (find_result)
                return std::make_pair(*find_result, dom_id);
        }
    }
    return {};
}

template < invocable_on_const_elements_r< bool > F, typename D, ExecutionPolicy_c ExecPolicy >
MeshPartition::cfind_result_t MeshPartition::find(F&& predicate, D&& domain_predicate, const ExecPolicy& policy) const
{
    return constifyFindResult(const_cast< MeshPartition* >(this)->find(
        std::forward< F >(predicate), std::forward< D >(domain_predicate), policy));
}

MeshPartition::find_result_t MeshPartition::find(el_id_t id)
{
    for (auto& [dom_id, dom] : domains)
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
    constexpr auto miss       = std::numeric_limits< el_ns_t >::max();
    constexpr auto el_dim     = ElementTraits< Element< T, O > >::native_dim;
    constexpr auto matchElDim = []< ElementTypes T_, el_o_t O_ >(const Element< T_, O_ >*) {
        return ElementTraits< Element< T_, O_ > >::native_dim - 1 == el_dim;
    };

    const auto boundary_nodes = getSortedArray(el.getNodes());
    const auto match_side     = [&]< ElementTypes T_, el_o_t O_ >(const Element< T_, O_ >* domain_element) {
        return detail::matchBoundaryNodesToElement(*domain_element, boundary_nodes);
    };

    const auto adjacent_elems = dual_graph->getElementAdjacent(el.getId());
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
    el_ns_t    side_index     = 0;

    const auto is_domain_element = [&]< ElementTypes T_, el_o_t O_ >(const Element< T_, O_ >& domain_element) {
        side_index = detail::matchBoundaryNodesToElement(domain_element, boundary_nodes);
        return side_index != std::numeric_limits< el_ns_t >::max();
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

BoundaryView MeshPartition::getBoundaryView(d_id_t boundary_id) const
{
    BoundaryView::boundary_element_view_variant_vector_t boundary_elements;
    boundary_elements.reserve(getDomainView(boundary_id).getNElements());

    const auto insert_boundary_element_view = [&](const auto& boundary_element) {
        const auto& [domain_element_variant_opt, side_index] = getElementBoundaryView(boundary_element, boundary_id);
        if (not domain_element_variant_opt)
            throw std::logic_error{"BoundaryView could not be constructed because some of the boundary elements are "
                                   "not edges/faces of any of the domain elements in the specified partition"};

        const auto emplace_element =
            [&, side_index = side_index]< ElementTypes T, el_o_t O >(const Element< T, O >* domain_element_ptr) {
                boundary_elements.emplace_back(
                    std::in_place_type< BoundaryElementView< T, O > >, *domain_element_ptr, side_index);
            };
        std::visit(emplace_element, domain_element_variant_opt->first);
    };

    cvisit(insert_boundary_element_view, {boundary_id});
    return BoundaryView{std::move(boundary_elements)};
}

size_t MeshPartition::getNElements() const
{
    return std::accumulate(
        domains.cbegin(), domains.cend(), 0, [](size_t s, const auto& d) { return s + d.second.getNElements(); });
}

std::vector< d_id_t > MeshPartition::getDomainIds() const
{
    std::vector< d_id_t > retval;
    retval.reserve(domains.size());
    std::ranges::transform(domains, back_inserter(retval), [](const auto& pair) { return pair.first; });
    return retval;
}

template < el_o_t O >
MeshPartition::domain_map_t MeshPartition::getConversionAlloc() const
{
    domain_map_t retval;
    for (const auto& [id, dom] : domains)
        retval.emplace_hint(retval.cend(), id, dom.template getConversionAlloc< O >());
    return retval;
}

MeshPartition::MeshPartition(MeshPartition::domain_map_t domains_) : domains{std::move(domains_)}
{
    constexpr size_t n_nodes_estimate_factor = 4; // TODO: come up with better heuristic
    nodes.reserve(getNElements() * n_nodes_estimate_factor);
    visit(
        [&](const auto& element) { std::ranges::for_each(element.getNodes(), [&](n_id_t n) { nodes.push_back(n); }); });
    std::ranges::sort(nodes);
    const auto range_to_erase = std::ranges::unique(nodes);
    nodes.erase(begin(range_to_erase), end(range_to_erase));
    nodes.shrink_to_fit();
}

auto MeshPartition::convertToMetisFormat() const
{
    std::vector< idx_t > eind, eptr;
    size_t               topo_size = 0;
    cvisit([&](const auto& element) { topo_size += element.getNodes().size(); });
    eind.reserve(topo_size);
    eptr.reserve(getNElements() + 1);
    eptr.push_back(0);
    for (el_id_t id = 0; id < getNElements(); ++id)
    {
        const auto el_ptr = find(id)->first;
        std::visit(
            [&]< ElementTypes T, el_o_t O >(const Element< T, O >* element) {
                std::ranges::for_each(element->getNodes(), [&](auto n) { eind.push_back(n); });
                eptr.push_back(eptr.back() + element->getNodes().size());
            },
            el_ptr);
    };
    return std::make_pair(std::move(eptr), std::move(eind));
}

MetisGraphWrapper MeshPartition::makeMetisDualGraph() const
{
    auto mesh_in_metis_format = convertToMetisFormat();
    auto& [eptr, eind]        = mesh_in_metis_format;

    auto  ne      = static_cast< idx_t >(getNElements());
    auto  nn      = static_cast< idx_t >(getNodes().size());
    idx_t ncommon = 2;
    idx_t numflag = 0;

    idx_t* xadj;
    idx_t* adjncy;

    const auto error = METIS_MeshToDual(&ne, &nn, eptr.data(), eind.data(), &ncommon, &numflag, &xadj, &adjncy);
    detail::handleMetisErrorCode(error);

    return MetisGraphWrapper{xadj, adjncy, getNElements()};
}
} // namespace lstr
#endif // L3STER_MESH_MESHPARTITION_HPP
