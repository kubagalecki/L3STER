#ifndef L3STER_MESH_MESHPARTITION_HPP
#define L3STER_MESH_MESHPARTITION_HPP

#include "mesh/BoundaryView.hpp"
#include "mesh/Domain.hpp"

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
} // namespace detail

class MeshPartition
{
public:
    using domain_map_t = std::map< d_id_t, Domain >;
    using node_vec_t   = std::vector< n_id_t >;

    MeshPartition()                         = default;
    MeshPartition(const MeshPartition&)     = delete;
    MeshPartition(MeshPartition&&) noexcept = default;
    MeshPartition& operator=(const MeshPartition&) = delete;
    MeshPartition& operator=(MeshPartition&&) noexcept = default;
    ~MeshPartition()                                   = default;

    inline explicit MeshPartition(domain_map_t domains_);
    MeshPartition(domain_map_t domains_, node_vec_t nodes_, node_vec_t ghost_nodes_)
        : domains{std::move(domains_)}, nodes{std::move(nodes_)}, ghost_nodes{std::move(ghost_nodes_)}
    {}

    template < invocable_on_elements F, detail::domain_predicate D >
    decltype(auto) visit(F&& element_visitor, D&& domain_predicate);
    template < invocable_on_const_elements F, detail::domain_predicate D >
    decltype(auto) cvisit(F&& element_visitor, D&& domain_predicate) const;
    template < invocable_on_elements_and< const DomainView > F, detail::domain_predicate D >
    decltype(auto) visit(F&& element_visitor, D&& domain_predicate);
    template < invocable_on_const_elements_and< const DomainView > F, detail::domain_predicate D >
    decltype(auto) cvisit(F&& element_visitor, D&& domain_predicate) const;
    template < invocable_on_elements F >
    decltype(auto) visit(F&& element_visitor);
    template < invocable_on_const_elements F >
    decltype(auto) cvisit(F&& element_visitor) const;
    template < invocable_on_elements_and< const DomainView > F >
    decltype(auto) visit(F&& element_visitor);
    template < invocable_on_const_elements_and< const DomainView > F >
    decltype(auto) cvisit(F&& element_visitor) const;
    template < invocable_on_elements F >
    decltype(auto) visit(F&& element_visitor, const std::vector< d_id_t >& domain_ids);
    template < invocable_on_const_elements F >
    decltype(auto) cvisit(F&& element_visitor, const std::vector< d_id_t >& domain_ids) const;
    template < invocable_on_elements_and< const DomainView > F >
    decltype(auto) visit(F&& element_visitor, const std::vector< d_id_t >& domain_ids);
    template < invocable_on_const_elements_and< const DomainView > F >
    decltype(auto) cvisit(F&& element_visitor, const std::vector< d_id_t >& domain_ids) const;

    // Note on finding elements: Elements are not ordered; if the predicate returns true for multiple elements, the
    // reference to any one of them may be returned
    using opt_el_ref  = std::optional< element_ptr_variant_t >;
    using opt_el_cref = std::optional< element_cptr_variant_t >;
    template < invocable_on_const_elements_r< bool > F >
    [[nodiscard]] opt_el_ref find(const F& predicate);
    template < invocable_on_const_elements_r< bool > F >
    [[nodiscard]] opt_el_cref find(const F& predicate) const;
    template < invocable_on_const_elements_r< bool > F >
    [[nodiscard]] opt_el_ref find(const F& predicate, const std::vector< d_id_t >& domain_ids);
    template < invocable_on_const_elements_r< bool > F >
    [[nodiscard]] opt_el_cref find(const F& predicate, const std::vector< d_id_t >& domain_ids) const;
    template < invocable_on_const_elements_r< bool > F, typename D >
    [[nodiscard]] opt_el_ref find(const F& predicate, const D& domain_predicate);
    template < invocable_on_const_elements_r< bool > F, typename D >
    [[nodiscard]] opt_el_cref find(const F& predicate, const D& domain_predicate) const;

    template < ElementTypes T, el_o_t O >
    [[nodiscard]] auto                getElementBoundaryView(const Element< T, O >& el, d_id_t d) const;
    [[nodiscard]] inline BoundaryView getBoundaryView(d_id_t) const;

    [[nodiscard]] DomainView    getDomainView(d_id_t id) const { return DomainView(domains.at(id), id); }
    [[nodiscard]] inline size_t getNElements() const;

    [[nodiscard]] const node_vec_t& getNodes() const noexcept { return nodes; }
    [[nodiscard]] const node_vec_t& getGhostNodes() const noexcept { return ghost_nodes; }

private:
    domain_map_t domains;
    node_vec_t   nodes{};
    node_vec_t   ghost_nodes{};
};

template < invocable_on_elements F, detail::domain_predicate D >
decltype(auto) MeshPartition::visit(F&& element_visitor, D&& domain_predicate)
{
    std::ranges::for_each(domains, [&](domain_map_t::value_type& domain) {
        if (domain_predicate(DomainView{domain.second, domain.first}))
            domain.second.visit(element_visitor);
    });
    return std::forward< F >(element_visitor);
}

template < invocable_on_const_elements F, detail::domain_predicate D >
decltype(auto) MeshPartition::cvisit(F&& element_visitor, D&& domain_predicate) const
{
    std::for_each(domains.cbegin(), domains.cend(), [&](const domain_map_t::value_type& domain) {
        if (domain_predicate(DomainView{domain.second, domain.first}))
            domain.second.cvisit(element_visitor);
    });
    return std::forward< F >(element_visitor);
}

template < invocable_on_elements_and< const DomainView > F, detail::domain_predicate D >
decltype(auto) MeshPartition::visit(F&& element_visitor, D&& domain_predicate)
{
    Domain     dummy;
    DomainView current_view{dummy, 0};
    visit([&](auto& element) { element_visitor(element, current_view); },
          [&](const DomainView& d) {
              current_view = d;
              return domain_predicate(d);
          });
    return std::forward< F >(element_visitor);
}

template < invocable_on_const_elements_and< const DomainView > F, detail::domain_predicate D >
decltype(auto) MeshPartition::cvisit(F&& element_visitor, D&& domain_predicate) const
{
    Domain     dummy;
    DomainView current_view{dummy, 0};
    cvisit([&](const auto& element) { element_visitor(element, current_view); },
           [&](const DomainView& d) {
               current_view = d;
               return domain_predicate(d);
           });
    return std::forward< F >(element_visitor);
}

template < invocable_on_elements F >
decltype(auto) MeshPartition::visit(F&& element_visitor)
{
    return visit(std::forward< F >(element_visitor), [](const DomainView&) { return true; });
}

template < invocable_on_const_elements F >
decltype(auto) MeshPartition::cvisit(F&& element_visitor) const
{
    return cvisit(std::forward< F >(element_visitor), [](const DomainView&) { return true; });
}

template < invocable_on_elements_and< const DomainView > F >
decltype(auto) MeshPartition::visit(F&& element_visitor)
{
    return visit(std::forward< F >(element_visitor), [](const DomainView&) { return true; });
}

template < invocable_on_const_elements_and< const DomainView > F >
decltype(auto) MeshPartition::cvisit(F&& element_visitor) const
{
    return cvisit(std::forward< F >(element_visitor), [](const DomainView&) { return true; });
}

template < invocable_on_elements F >
decltype(auto) MeshPartition::visit(F&& element_visitor, const std::vector< d_id_t >& domain_ids)
{
    const auto domain_predicate = [&domain_ids](const DomainView& d) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&d](const d_id_t& d1) { return d.getID() == d1; });
    };
    return visit(std::forward< F >(element_visitor), domain_predicate);
}

template < invocable_on_const_elements F >
decltype(auto) MeshPartition::cvisit(F&& element_visitor, const std::vector< d_id_t >& domain_ids) const
{
    const auto domain_predicate = [&](const DomainView& d) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&](const d_id_t& d1) { return d.getID() == d1; });
    };
    return cvisit(std::forward< F >(element_visitor), domain_predicate);
}

template < invocable_on_elements_and< const DomainView > F >
decltype(auto) MeshPartition::visit(F&& element_visitor, const std::vector< d_id_t >& domain_ids)
{
    const auto domain_predicate = [&domain_ids](const DomainView& d) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&d](const d_id_t& d1) { return d.getID() == d1; });
    };
    return visit(std::forward< F >(element_visitor), domain_predicate);
}

template < invocable_on_const_elements_and< const DomainView > F >
decltype(auto) MeshPartition::cvisit(F&& element_visitor, const std::vector< d_id_t >& domain_ids) const
{
    const auto domain_predicate = [&](const DomainView& d) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&](const d_id_t& d1) { return d.getID() == d1; });
    };
    return cvisit(std::forward< F >(element_visitor), domain_predicate);
}

template < invocable_on_const_elements_r< bool > F >
std::optional< element_ptr_variant_t > MeshPartition::find(const F& predicate)
{
    return find(predicate, [](const DomainView&) { return true; });
}

template < invocable_on_const_elements_r< bool > F >
std::optional< element_cptr_variant_t > MeshPartition::find(const F& predicate) const
{
    return find(predicate, [](const DomainView&) { return true; });
}

template < invocable_on_const_elements_r< bool > F >
std::optional< element_ptr_variant_t > MeshPartition::find(const F& predicate, const std::vector< d_id_t >& domain_ids)
{
    return find(predicate, [&](const auto& domain_view) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&](d_id_t d) { return d == domain_view.getID(); });
    });
}

template < invocable_on_const_elements_r< bool > F >
std::optional< element_cptr_variant_t > MeshPartition::find(const F&                     predicate,
                                                            const std::vector< d_id_t >& domain_ids) const
{
    return find(predicate, [&](const auto& domain_view) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&](d_id_t d) { return d == domain_view.getID(); });
    });
}

template < invocable_on_const_elements_r< bool > F, typename D >
std::optional< element_ptr_variant_t > MeshPartition::find(const F& predicate, const D& domain_predicate)
{
    std::optional< element_ptr_variant_t > ret_val;
    for (auto& domain_map_entry : domains)
    {
        if (domain_predicate(DomainView{domain_map_entry.second, domain_map_entry.first}))
        {
            ret_val = domain_map_entry.second.findElement(predicate);
            if (ret_val)
                break;
        }
    }
    return ret_val;
}

template < invocable_on_const_elements_r< bool > F, typename D >
std::optional< element_cptr_variant_t > MeshPartition::find(const F& predicate, const D& domain_predicate) const
{
    std::optional< element_cptr_variant_t > ret_val;
    for (const auto& domain_map_entry : domains)
    {
        if (domain_predicate(DomainView{domain_map_entry.second, domain_map_entry.first}))
        {
            ret_val = domain_map_entry.second.findElement(predicate);
            if (ret_val)
                break;
        }
    }
    return ret_val;
}

namespace detail
{
template < typename Element, size_t N, el_ns_t I >
consteval auto makeSideMatcher()
{
    return [](const Element& element, const std::array< n_id_t, N >& sorted_side_nodes) {
        constexpr auto& side_inds = std::get< I >(ElementTraits< Element >::boundary_table);
        if constexpr (std::tuple_size_v< std::decay_t< decltype(side_inds) > > == N)
        {
            std::array< n_id_t, N > element_side_nodes{};
            std::transform(side_inds.cbegin(),
                           side_inds.cend(),
                           element_side_nodes.begin(),
                           [&element_nodes = element.getNodes()](el_locind_t i) { return element_nodes[i]; });
            std::ranges::sort(element_side_nodes);
            return std::equal(element_side_nodes.cbegin(), element_side_nodes.cend(), sorted_side_nodes.cbegin());
        }
        else
            return false;
    };
}

template < typename Element, size_t N, el_ns_t I >
struct SideMatcher
{
    static consteval auto get()
    {
        return [](const Element& element, const std::array< n_id_t, N >& sorted_side_nodes) {
            if (makeSideMatcher< Element, N, I >()(element, sorted_side_nodes))
                return I;
            else
                return SideMatcher< Element, N, I - 1 >::get()(element, sorted_side_nodes);
        };
    }
};

template < typename Element, size_t N >
struct SideMatcher< Element, N, 0 >
{
    static consteval auto get()
    {
        return [](const Element& element, const std::array< n_id_t, N >& sorted_side_nodes) {
            if (makeSideMatcher< Element, N, 0 >()(element, sorted_side_nodes))
                return static_cast< el_ns_t >(0u);
            else
                return std::numeric_limits< el_ns_t >::max();
        };
    }
};
} // namespace detail

template < ElementTypes T, el_o_t O >
auto MeshPartition::getElementBoundaryView(const Element< T, O >& el, d_id_t d) const
{
    const auto boundary_nodes = [&]() {
        auto bn = el.getNodes();
        std::ranges::sort(bn);
        return bn;
    }();
    constexpr size_t boundary_size = std::tuple_size_v< std::decay_t< decltype(boundary_nodes) > >;

    el_ns_t    side_index        = 0;
    const auto is_domain_element = [&]< ElementTypes DT, el_o_t DO >(const Element< DT, DO >& domain_element) {
        constexpr auto n_sides = ElementTraits< Element< DT, DO > >::n_sides;
        constexpr auto matcher = detail::SideMatcher< Element< DT, DO >, boundary_size, n_sides - 1 >::get();

        side_index = matcher(domain_element, boundary_nodes);
        return side_index != std::numeric_limits< el_ns_t >::max();
    };
    return std::make_pair(
        find(is_domain_element, [&](const DomainView& dv) { return dv.getDim() == getDomainView(d).getDim() + 1; }),
        side_index);
}

BoundaryView MeshPartition::getBoundaryView(d_id_t boundary_id) const
{
    BoundaryView::boundary_element_view_variant_vector_t boundary_elements;
    boundary_elements.reserve(getDomainView(boundary_id).getNElements());

    const auto insert_boundary_element_view = [&](const auto& boundary_element) {
        const auto& [domain_element_variant_opt, side_index] = getElementBoundaryView(boundary_element, boundary_id);
        if (!domain_element_variant_opt)
            throw std::logic_error{"BoundaryView could not be constructed because some of the boundary elements are "
                                   "not edges/faces of any of the domain elements in the specified partition"};

        const auto emplace_element =
            [&, side_index = side_index]< ElementTypes T, el_o_t O >(const Element< T, O >* domain_element_ptr) {
                boundary_elements.emplace_back(
                    std::in_place_type< BoundaryElementView< T, O > >, *domain_element_ptr, side_index);
            };
        std::visit(emplace_element, *domain_element_variant_opt);
    };

    cvisit(insert_boundary_element_view, {boundary_id});
    return BoundaryView{std::move(boundary_elements)};
}

size_t MeshPartition::getNElements() const
{
    return std::accumulate(
        domains.cbegin(), domains.cend(), 0, [](size_t s, const auto& d) { return s + d.second.getNElements(); });
}

MeshPartition::MeshPartition(MeshPartition::domain_map_t domains_) : domains{std::move(domains_)}
{
    constexpr size_t n_nodes_estimate_factor = 4;
    nodes.reserve(getNElements() * n_nodes_estimate_factor);
    visit(
        [&](const auto& element) { std::ranges::for_each(element.getNodes(), [&](n_id_t n) { nodes.push_back(n); }); });
    std::ranges::sort(nodes);
    nodes.erase(std::ranges::unique(nodes).begin(), nodes.end());
    nodes.shrink_to_fit();
}
} // namespace lstr
#endif // L3STER_MESH_MESHPARTITION_HPP
