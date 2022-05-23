#ifndef L3STER_MESH_BOUNDARYVIEW_HPP
#define L3STER_MESH_BOUNDARYVIEW_HPP

#include "l3ster/mesh/Aliases.hpp"
#include "l3ster/mesh/BoundaryElementView.hpp"

#include <utility>

namespace lstr
{
class BoundaryView
{
public:
    using boundary_element_view_variant_t =
        parametrize_type_over_element_types_and_orders_t< std::variant, BoundaryElementView >;
    using boundary_element_view_variant_vector_t = std::vector< boundary_element_view_variant_t >;

    explicit BoundaryView(boundary_element_view_variant_vector_t in) : boundary_elements{std::move(in)} {};

    template < invocable_on_const_elements F >
    decltype(auto) visit(F&& visitor) const;

    [[nodiscard]] size_t size() const { return boundary_elements.size(); }

private:
    const boundary_element_view_variant_vector_t boundary_elements{};
};

template < invocable_on_const_elements F >
decltype(auto) BoundaryView::visit(F&& visitor) const
{
    std::ranges::for_each(boundary_elements, [&](const auto& boundary_el) { std::visit(visitor, boundary_el); });
    return std::forward< F >(visitor);
}
} // namespace lstr
#endif // L3STER_MESH_BOUNDARYVIEW_HPP
