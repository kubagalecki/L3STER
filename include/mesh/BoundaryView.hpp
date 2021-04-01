#ifndef L3STER_MESH_BOUNDARYVIEW_HPP
#define L3STER_MESH_BOUNDARYVIEW_HPP

#include <utility>

#include "mesh/BoundaryElementView.hpp"

namespace lstr
{
class BoundaryView
{
public:
    using boundary_element_view_variant_t =
        parametrize_type_over_element_types_and_orders_t< std::variant, BoundaryElementView >;
    using boundary_element_view_variant_vector_t = std::vector< boundary_element_view_variant_t >;

    explicit BoundaryView(boundary_element_view_variant_vector_t in) : boundary_elements{std::move(in)} {};

    template < typename F >
    decltype(auto) visit(F&& visitor) const;

    [[nodiscard]] size_t size() const { return boundary_elements.size(); }

private:
    boundary_element_view_variant_vector_t boundary_elements{};
};

template < typename F >
decltype(auto) BoundaryView::visit(F&& visitor) const
{
    std::for_each(boundary_elements.cbegin(), boundary_elements.cend(), [&](const auto& boundary) {
        std::visit< void >(visitor, boundary);
    });
    return std::forward< F >(visitor);
}
} // namespace lstr

#endif // L3STER_MESH_BOUNDARYVIEW_HPP
