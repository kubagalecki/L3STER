#ifndef L3STER_MESH_MESHPARTITION_HPP
#define L3STER_MESH_MESHPARTITION_HPP

#include "mesh/BoundaryView.hpp"
#include "mesh/Domain.hpp"

#include <map>
#include <variant>

namespace lstr::mesh
{
class MeshPartition
{
public:
    using domain_map_t = std::map< types::d_id_t, Domain >;

    MeshPartition()                     = default;
    MeshPartition(const MeshPartition&) = delete;
    MeshPartition(MeshPartition&&)      = default;
    MeshPartition& operator=(const MeshPartition&) = delete;
    MeshPartition& operator=(MeshPartition&&) = default;
    ~MeshPartition()                          = default;

    explicit MeshPartition(MeshPartition::domain_map_t domains_) : domains{std::move(domains_)} {}

    // Visit domains
    template < typename F >
    auto visitAllElements(F&& element_visitor);

    template < typename F >
    auto cvisitAllElements(F&& element_visitor) const;

    template < typename F >
    auto visitSpecifiedDomains(F&& element_visitor, const std::vector< types::d_id_t >& domain_ids);

    template < typename F >
    auto cvisitSpecifiedDomains(F&&                                 element_visitor,
                                const std::vector< types::d_id_t >& domain_ids) const;

    template < typename F, typename D >
    auto visitDomainIf(F&& element_visitor, D&& domain_predicate);

    template < typename F, typename D >
    auto cvisitDomainIf(F&& element_visitor, D&& domain_predicate) const;

    // Find element
    // Note: Elements are not ordered; if predicate returns true for multiple elements, the
    // reference to any one of them may be returned
    template < typename F >
    std::optional< element_ref_variant_t > findElement(const F& predicate);

    template < typename F >
    std::optional< element_cref_variant_t > findElement(const F& predicate) const;

    template < typename F >
    std::optional< element_ref_variant_t >
    findElementInSpecifiedDomains(const F&                            predicate,
                                  const std::vector< types::d_id_t >& domain_ids);

    template < typename F >
    std::optional< element_cref_variant_t >
    findElementInSpecifiedDomains(const F&                            predicate,
                                  const std::vector< types::d_id_t >& domain_ids) const;

    template < typename F, typename D >
    std::optional< element_ref_variant_t > findElementIfDomain(const F& predicate,
                                                               const D& domain_predicate);

    template < typename F, typename D >
    std::optional< element_cref_variant_t > findElementIfDomain(const F& predicate,
                                                                const D& domain_predicate) const;

    [[nodiscard]] DomainView getDomainView(types::d_id_t id) const
    {
        return DomainView(domains.at(id), id);
    }

    inline void pushDomain(types::d_id_t, Domain);
    inline void popDomain(types::d_id_t);

    [[nodiscard]] inline BoundaryView getBoundaryView(const types::d_id_t&) const;

private:
    domain_map_t domains;
};

template < typename F >
auto MeshPartition::visitAllElements(F&& element_visitor)
{
    return visitDomainIf(std::forward< F >(element_visitor),
                         [](const DomainView&) { return true; });
}

template < typename F >
auto MeshPartition::cvisitAllElements(F&& element_visitor) const
{
    return cvisitDomainIf(std::forward< F >(element_visitor),
                          [](const DomainView&) { return true; });
}

template < typename F >
auto MeshPartition::visitSpecifiedDomains(F&&                                 element_visitor,
                                          const std::vector< types::d_id_t >& domain_ids)
{
    auto domain_predicate = [&domain_ids](const DomainView& d) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&d](const types::d_id_t& d1) {
            return d.getID() == d1;
        });
    };

    return visitDomainIf(std::forward< F >(element_visitor), std::move(domain_predicate));
}

template < typename F >
auto MeshPartition::cvisitSpecifiedDomains(F&&                                 element_visitor,
                                           const std::vector< types::d_id_t >& domain_ids) const
{
    auto domain_predicate = [&domain_ids](const DomainView& d) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&d](const types::d_id_t& d1) {
            return d.getID() == d1;
        });
    };

    return cvisitDomainIf(std::forward< F >(element_visitor), std::move(domain_predicate));
}

template < typename F, typename D >
auto MeshPartition::visitDomainIf(F&& element_visitor, D&& domain_predicate)
{
    static_assert(std::is_invocable_r_v< bool, D, DomainView& >);

    auto visitor   = std::forward< F >(element_visitor);
    auto predicate = std::forward< D >(domain_predicate);

    const auto domain_visitor = [&visitor](domain_map_t::value_type& domain) {
        domain.second.visit(visitor);
    };

    std::for_each(domains.begin(),
                  domains.end(),
                  [&domain_visitor, &predicate](domain_map_t::value_type& domain) {
                      if (predicate(DomainView{domain.second, domain.first}))
                          domain_visitor(domain);
                  });

    return visitor;
}

template < typename F, typename D >
auto MeshPartition::cvisitDomainIf(F&& element_visitor, D&& domain_predicate) const
{
    static_assert(std::is_invocable_r_v< bool, D, DomainView& >);

    auto visitor   = std::forward< F >(element_visitor);
    auto predicate = std::forward< D >(domain_predicate);

    const auto domain_visitor = [&visitor](const domain_map_t::value_type& domain) {
        domain.second.cvisit(visitor);
    };

    std::for_each(domains.cbegin(),
                  domains.cend(),
                  [&domain_visitor, &predicate](const domain_map_t::value_type& domain) {
                      if (predicate(DomainView{domain.second, domain.first}))
                          domain_visitor(domain);
                  });

    return visitor;
}

template < typename F >
std::optional< element_ref_variant_t > MeshPartition::findElement(const F& predicate)
{
    return findElementIfDomain(predicate, [](const auto&) { return true; });
}

template < typename F >
std::optional< element_cref_variant_t > MeshPartition::findElement(const F& predicate) const
{
    return findElementIfDomain(predicate, [](const auto&) { return true; });
}

template < typename F >
std::optional< element_ref_variant_t >
MeshPartition::findElementInSpecifiedDomains(const F&                            predicate,
                                             const std::vector< types::d_id_t >& domain_ids)
{
    return findElementIfDomain(predicate, [&domain_ids](const auto& domain_view) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&domain_view](types::d_id_t d) {
            return d == domain_view.getID();
        });
    });
}

template < typename F >
std::optional< element_cref_variant_t >
MeshPartition::findElementInSpecifiedDomains(const F&                            predicate,
                                             const std::vector< types::d_id_t >& domain_ids) const
{
    return findElementIfDomain(predicate, [&domain_ids](const auto& domain_view) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&domain_view](types::d_id_t d) {
            return d == domain_view.getID();
        });
    });
}

template < typename F, typename D >
std::optional< element_ref_variant_t > MeshPartition::findElementIfDomain(const F& predicate,
                                                                          const D& domain_predicate)
{
    std::optional< element_ref_variant_t > ret_val;

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

template < typename F, typename D >
std::optional< element_cref_variant_t >
MeshPartition::findElementIfDomain(const F& predicate, const D& domain_predicate) const
{
    std::optional< element_cref_variant_t > ret_val;

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

void MeshPartition::pushDomain(types::d_id_t id, Domain d)
{
    domains[id] = std::move(d);
}

void MeshPartition::popDomain(types::d_id_t id)
{
    domains.erase(id);
}

namespace detail
{
template < typename Element, size_t N, types::el_ns_t I >
consteval auto makeSideMatcher()
{
    return [](const Element& element, const std::array< types::n_id_t, N >& sorted_side_nodes) {
        constexpr auto& side_inds = std::get< I >(ElementTraits< Element >::boundary_table);
        if constexpr (std::tuple_size_v< std::decay_t< decltype(side_inds) > > == N)
        {
            std::array< types::n_id_t, N > element_side_nodes;
            std::transform(side_inds.cbegin(),
                           side_inds.cend(),
                           element_side_nodes.begin(),
                           [&element_nodes = element.getNodes()](types::el_locind_t i) {
                               return element_nodes[i];
                           });

            std::sort(element_side_nodes.begin(), element_side_nodes.end());

            return std::equal(
                element_side_nodes.cbegin(), element_side_nodes.cend(), sorted_side_nodes.cbegin());
        }
        else
            return false;
    };
}

template < typename Element, size_t N, types::el_ns_t I >
struct sideMatcher
{
    static consteval auto get()
    {
        return [](const Element& element, const std::array< types::n_id_t, N >& sorted_side_nodes) {
            if (makeSideMatcher< Element, N, I >()(element, sorted_side_nodes))
                return I;
            else
                return sideMatcher< Element, N, I - 1 >::get()(element, sorted_side_nodes);
        };
    }
};

template < typename Element, size_t N >
struct sideMatcher< Element, N, 0 >
{
    static consteval auto get()
    {
        return [](const Element& element, const std::array< types::n_id_t, N >& sorted_side_nodes) {
            if (makeSideMatcher< Element, N, 0 >()(element, sorted_side_nodes))
                return static_cast< types::el_ns_t >(0u);
            else
                return std::numeric_limits< types::el_ns_t >::max();
        };
    }
};
} // namespace detail

BoundaryView MeshPartition::getBoundaryView(const types::d_id_t& boundary_id) const
{
    const auto boundary_domain_view = getDomainView(boundary_id);
    const auto boundary_dim         = boundary_domain_view.getDim();
    const auto n_boundary_parts     = boundary_domain_view.getNElements();

    BoundaryView::boundary_element_view_variant_vector_t boundary_elements;
    boundary_elements.reserve(n_boundary_parts);

    const auto insert_boundary_element_view = [this, &boundary_dim, &boundary_elements](
                                                  const auto& boundary_element) {
        const auto boundary_nodes = [bn = boundary_element.getNodes()]() mutable {
            std::sort(bn.begin(), bn.end());
            return bn;
        }();

        types::el_ns_t side_index = 0;

        const auto is_domain_element =
            [&side_index, this, &boundary_nodes](const auto& domain_element) {
                constexpr size_t boundary_size =
                    std::tuple_size_v< std::decay_t< decltype(boundary_nodes) > >;
                using domain_element_t = std::decay_t< decltype(domain_element) >;
                constexpr auto n_sides = ElementTraits< domain_element_t >::n_sides;

                constexpr auto matcher =
                    detail::sideMatcher< domain_element_t, boundary_size, n_sides - 1 >::get();

                const auto matched_side = matcher(domain_element, boundary_nodes);
                side_index              = matched_side;
                return matched_side != std::numeric_limits< types::el_ns_t >::max();
            };

        const auto domain_element_variant_opt =
            findElementIfDomain(is_domain_element, [&boundary_dim](const DomainView& d) {
                return d.getDim() == boundary_dim + 1;
            });

        if (!domain_element_variant_opt)
            throw std::logic_error{
                "BoundaryView could not be constructed because some of the boundary elements are "
                "not edges/faces of any of the domain elements in the specified partition"};

        const auto emplace_element =
            [this, &side_index, &boundary_elements](const auto& domain_element_ref) {
                using element_t =
                    std::decay_t< typename std::decay_t< decltype(domain_element_ref) >::type >;
                using element_traits_t   = ElementTraits< element_t >;
                using boundary_element_t = BoundaryElementView< element_traits_t::element_type,
                                                                element_traits_t::element_order >;

                boundary_elements.emplace_back(
                    std::in_place_type< boundary_element_t >, domain_element_ref, side_index);
            };
        std::visit(emplace_element, *domain_element_variant_opt);
    };

    cvisitSpecifiedDomains(insert_boundary_element_view, {boundary_id});
    return BoundaryView{std::move(boundary_elements)};
}

} // namespace lstr::mesh

#endif // L3STER_MESH_MESHPARTITION_HPP
