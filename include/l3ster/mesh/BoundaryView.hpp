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

    BoundaryView() = default;
    explicit BoundaryView(boundary_element_view_variant_vector_t in) : boundary_elements{std::move(in)} {};

    template < invocable_on_boundary_element_views F, ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(F&& visitor, ExecPolicy&& policy = {}) const
    {
        if constexpr (std::same_as< std::remove_cvref_t< ExecPolicy >, std::execution::sequenced_policy >)
        {
            std::ranges::for_each(boundary_elements,
                                  [&visitor](const auto& boundary_el) { std::visit(visitor, boundary_el); });
        }
        else
            std::for_each(std::forward< ExecPolicy >(policy),
                          begin(boundary_elements),
                          end(boundary_elements),
                          [&visitor](const auto& boundary_el) { std::visit(visitor, boundary_el); });
    }

    template < typename Zero,
               typename Proj,
               typename Red,
               ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    Zero reduce(Zero&& zero, Proj&& projection, Red&& reduction, ExecPolicy&& policy = {}) const
        requires invocable_on_boundary_element_views_r< Proj, Zero > and
                 requires(std::remove_cvref_t< Zero > z, std::remove_cvref_t< Red > r) {
                     {
                         std::invoke(
                             std::forward< Red >(reduction), std::forward< Zero >(zero), std::forward< Zero >(zero))
                         } -> std::convertible_to< std::remove_cvref_t< Zero > >;
                 }
    {
        return std::transform_reduce(
            std::forward< ExecPolicy >(policy),
            begin(boundary_elements),
            end(boundary_elements),
            std::forward< Zero >(zero),
            std::forward< Red >(reduction),
            [&](const boundary_element_view_variant_t& var) { return std::visit< Zero >(projection, var); });
    }

    [[nodiscard]] size_t size() const { return boundary_elements.size(); }

private:
    const boundary_element_view_variant_vector_t boundary_elements{};
};
} // namespace lstr
#endif // L3STER_MESH_BOUNDARYVIEW_HPP
