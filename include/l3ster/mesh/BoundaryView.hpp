#ifndef L3STER_MESH_BOUNDARYVIEW_HPP
#define L3STER_MESH_BOUNDARYVIEW_HPP

#include "l3ster/mesh/BoundaryElementView.hpp"
#include "l3ster/mesh/Domain.hpp"

#include <utility>

namespace lstr::mesh
{
template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
class MeshPartition;

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
class BoundaryView
{
    using Constraint = ElementDeductionHelper< orders... >;

public:
    using boundary_element_view_variant_t =
        parametrize_type_over_element_types_and_orders_t< std::variant, BoundaryElementView, orders... >;
    using boundary_element_view_variant_vector_t = std::vector< boundary_element_view_variant_t >;

    BoundaryView() = default;
    BoundaryView(boundary_element_view_variant_vector_t in, const MeshPartition< orders... >& parent)
        : m_boundary_elements{std::move(in)}, m_parent_partition{std::addressof(parent)} {};

    template < typename F, ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(F&& visitor, ExecPolicy&& policy = {}) const
        requires(Constraint::template invocable_on_boundary_views< F >);
    template < typename Zero,
               typename Proj,
               typename Red,
               ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    Zero reduce(Zero&& zero, Proj&& projection, Red&& reduction, ExecPolicy policy = {}) const
        requires(Constraint::template invocable_on_boundary_views_return< Zero, Proj > and
                 requires(std::remove_cvref_t< Zero > z, std::remove_cvref_t< Red > r) {
                     {
                         std::invoke(
                             std::forward< Red >(reduction), std::forward< Zero >(zero), std::forward< Zero >(zero))
                     } -> std::convertible_to< std::remove_cvref_t< Zero > >;
                 });

    [[nodiscard]] const MeshPartition< orders... >* getParent() const { return m_parent_partition; }
    [[nodiscard]] size_t                            size() const { return m_boundary_elements.size(); }

private:
    boundary_element_view_variant_vector_t m_boundary_elements;
    const MeshPartition< orders... >*      m_parent_partition{};
};

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < typename F, ExecutionPolicy_c ExecPolicy >
void BoundaryView< orders... >::visit(F&& visitor, ExecPolicy&& policy) const
    requires(Constraint::template invocable_on_boundary_views< F >)
{
    if constexpr (std::same_as< std::remove_cvref_t< ExecPolicy >, std::execution::sequenced_policy >)
    {
        std::ranges::for_each(m_boundary_elements,
                              [&visitor](const auto& boundary_el) { std::visit(visitor, boundary_el); });
    }
    else
        std::for_each(std::forward< ExecPolicy >(policy),
                      begin(m_boundary_elements),
                      end(m_boundary_elements),
                      [&visitor](const auto& boundary_el) { std::visit(visitor, boundary_el); });
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < typename Zero, typename Proj, typename Red, ExecutionPolicy_c ExecPolicy >
Zero BoundaryView< orders... >::reduce(Zero&& zero, Proj&& projection, Red&& reduction, ExecPolicy policy) const
    requires(Constraint::template invocable_on_boundary_views_return< Zero, Proj > and
             requires(std::remove_cvref_t< Zero > z, std::remove_cvref_t< Red > r) {
                 {
                     std::invoke(std::forward< Red >(reduction), std::forward< Zero >(zero), std::forward< Zero >(zero))
                 } -> std::convertible_to< std::remove_cvref_t< Zero > >;
             })
{
    return std::transform_reduce(
        policy,
        begin(m_boundary_elements),
        end(m_boundary_elements),
        std::forward< Zero >(zero),
        std::forward< Red >(reduction),
        [&](const boundary_element_view_variant_t& var) { return std::visit< Zero >(projection, var); });
}
} // namespace lstr::mesh
#endif // L3STER_MESH_BOUNDARYVIEW_HPP
