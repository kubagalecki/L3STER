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

    template < invocable_on_boundary_element_views F, ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    decltype(auto) visit(F&& visitor, ExecPolicy&& policy = {}) const
    {
        if constexpr (std::same_as< std::remove_cvref_t< ExecPolicy >, std::execution::sequenced_policy >)
        {
            std::ranges::for_each(boundary_elements,
                                  [&visitor](const auto& boundary_el) { std::visit(visitor, boundary_el); });
            return std::forward< F >(visitor);
        }
        else
            std::for_each(std::forward< ExecPolicy >(policy),
                          begin(boundary_elements),
                          end(boundary_elements),
                          [&visitor](const auto& boundary_el) { std::visit(visitor, boundary_el); });
    }

    [[nodiscard]] size_t size() const { return boundary_elements.size(); }

private:
    const boundary_element_view_variant_vector_t boundary_elements{};
};
} // namespace lstr
#endif // L3STER_MESH_BOUNDARYVIEW_HPP
