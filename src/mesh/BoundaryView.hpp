#ifndef L3STER_MESH_BOUNDARYVIEW_HPP
#define L3STER_MESH_BOUNDARYVIEW_HPP

namespace lstr::mesh
{
class BoundaryView
{
public:
    using boundary_element_view_variant_t =
        parametrize_type_over_element_types_and_orders_t< std::variant, BoundaryElementView >;
    using boundary_element_view_variant_vector_t = std::vector< boundary_element_view_variant_t >;

    BoundaryView()                        = delete;
    BoundaryView(const BoundaryView&)     = default;
    BoundaryView(BoundaryView&&) noexcept = default;
    BoundaryView& operator=(const BoundaryView&) = default;
    BoundaryView& operator=(BoundaryView&&) noexcept = default;
    inline BoundaryView(const MeshPartition& topology, const types::d_id_t& boundary_id);

    template < typename F >
    auto visit(F&& element_variant) const;

private:
    boundary_element_view_variant_vector_t boundary_element_view_variant_vector;
};

namespace helpers
{
template < typename Element, size_t N, types::el_ns_t I >
constexpr auto makeSideMatcher()
{
    return [](const Element& element, const std::array< types::n_id_t, N >& sorted_side_nodes) {
        constexpr auto& side_inds = ElementTraits< Element >::boundary_table[I];
        if constexpr (side_inds.front() == N)
        {
            std::array< types::n_id_t, N > element_side_nodes;
            std::transform(side_inds.cbegin() + 1,
                           side_inds.cbegin() + 1 + N,
                           element_side_nodes.begin(),
                           [&side_inds, &element_nodes = element.getNodes()](types::el_locind_t i) {
                               return element_nodes[i];
                           });

            std::sort(element_side_nodes.begin(), element_side_nodes.end());

            bool ret;
            ret = std::equal(
                element_side_nodes.cbegin(), element_side_nodes.cend(), sorted_side_nodes.cbegin());
            return ret;
        }
        else
            return false;
    };
}

template < typename Element, size_t N, types::el_ns_t I >
struct sideMatcher
{
    static constexpr auto get()
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
    static constexpr auto get()
    {
        return [](const Element& element, const std::array< types::n_id_t, N >& sorted_side_nodes) {
            if (makeSideMatcher< Element, N, 0 >()(element, sorted_side_nodes))
                return static_cast< types::el_ns_t >(0u);
            else
                return std::numeric_limits< types::el_ns_t >::max();
        };
    }
};
} // namespace helpers

inline BoundaryView::BoundaryView(const MeshPartition& topology, const types::d_id_t& boundary_id)
{
    const auto boundary_domain_view = topology.getDomainView(boundary_id);
    const auto boundary_dim         = boundary_domain_view.getDim();
    const auto n_boundary_parts     = boundary_domain_view.getNElements();

    boundary_element_view_variant_vector.reserve(n_boundary_parts);

    const auto insert_boundary_element_view = [&topology, &boundary_dim, this](
                                                  const auto& boundary_element) {
        const auto boundary_nodes = [bn = boundary_element.getNodes()]() mutable {
            std::sort(bn.begin(), bn.end());
            return bn;
        }();
        constexpr size_t boundary_size =
            std::tuple_size_v< std::decay_t< decltype(boundary_nodes) > >;

        types::el_ns_t side_index = 0;

        const auto is_domain_element =
            [&side_index, &topology, &boundary_nodes, &boundary_size](const auto& domain_element) {
                using domain_element_t = std::decay_t< decltype(domain_element) >;
                constexpr auto n_sides = ElementTraits< domain_element_t >::n_sides;

                constexpr auto matcher =
                    helpers::sideMatcher< domain_element_t, boundary_size, n_sides - 1 >::get();

                const auto matched_side = matcher(domain_element, boundary_nodes);
                side_index              = matched_side;
                return matched_side != std::numeric_limits< types::el_ns_t >::max();
            };

        const auto domain_element_variant_opt =
            topology.findElementIfDomain(is_domain_element, [&boundary_dim](const DomainView& d) {
                return d.getDim() == boundary_dim + 1;
            });

        if (!domain_element_variant_opt)
            throw std::logic_error{
                "BoundaryView could not be constructed because some of the boundary elements are "
                "not edges/faces of any of the domain elements in the specified partition"};

        const auto emplace_element = [this, &side_index](const auto& domain_element_ref) {
            using element_t =
                std::decay_t< typename std::decay_t< decltype(domain_element_ref) >::type >;
            constexpr auto element_type  = ElementTraits< element_t >::element_type;
            constexpr auto element_order = ElementTraits< element_t >::element_order;
            using boundary_element_t     = BoundaryElementView< element_type, element_order >;

            this->boundary_element_view_variant_vector.emplace_back(
                std::in_place_type< boundary_element_t >, domain_element_ref, side_index);
        };
        std::visit(emplace_element, *domain_element_variant_opt);
    };

    topology.cvisitSpecifiedDomains(insert_boundary_element_view, {boundary_id});
}
template < typename F >
auto BoundaryView::visit(F&& visitor) const
{
    // static_assert(is_invocable_on_all_elements_v< F >);

    auto       boundary_visitor = std::forward< F >(visitor);
    const auto wrapped_visitor  = [&boundary_visitor](const auto& element_variant) {
        std::visit(boundary_visitor, element_variant);
    };

    std::for_each(boundary_element_view_variant_vector.cbegin(),
                  boundary_element_view_variant_vector.cend(),
                  std::ref(wrapped_visitor));

    return boundary_visitor;
}
} // namespace lstr::mesh

#endif // L3STER_MESH_BOUNDARYVIEW_HPP
