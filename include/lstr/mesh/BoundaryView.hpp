#ifndef L3STER_MESH_BOUNDARYVIEW_HPP
#define L3STER_MESH_BOUNDARYVIEW_HPP

#include <utility>

#include "lstr/mesh/BoundaryElementView.hpp"

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
    explicit BoundaryView(boundary_element_view_variant_vector_t  in)
        : boundary_element_view_variant_vector{std::move(in)} {};

    template < typename F >
    auto visit(F&& visitor) const;

    [[nodiscard]] size_t size() const { return boundary_element_view_variant_vector.size(); }

private:
    boundary_element_view_variant_vector_t boundary_element_view_variant_vector;
};

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
