#ifndef L3STER_MESH_BOUNDARYVIEW_HPP
#define L3STER_MESH_BOUNDARYVIEW_HPP

#include "l3ster/mesh/BoundaryElementView.hpp"
#include "l3ster/util/ArrayOwner.hpp"
#include "l3ster/util/Ranges.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

#include <utility>

namespace lstr::mesh
{
namespace detail
{
template < el_o_t... orders >
using boundary_element_view_variant_t =
    parametrize_type_over_element_types_and_orders_t< std::variant, BoundaryElementView, orders... >;
template < el_o_t... orders >
using boundary_element_view_variant_array_t = util::ArrayOwner< boundary_element_view_variant_t< orders... > >;
} // namespace detail

template < typename Visitor, el_o_t... orders >
concept BoundaryViewVisitor_c = ElementDeductionHelper< orders... >::template invocable_on_boundary_views< Visitor >;

template < typename Zero, typename Transform, typename Reduction, el_o_t... orders >
concept BoundaryTransformReducible_c =
    ElementDeductionHelper< orders... >::template invocable_on_boundary_views_return< Zero, Transform > and
    ReductionFor_c< Reduction, Zero >;

template < el_o_t... orders >
class BoundaryView
{
    static_assert(sizeof...(orders) > 0);

public:
    BoundaryView(detail::boundary_element_view_variant_array_t< orders... > boundary_elements)
        : m_boundary_elements{std::move(boundary_elements)}
    {}

    template < BoundaryViewVisitor_c< orders... > Visitor, ExecutionPolicy_c ExecPolicy >
    void visit(Visitor&& visitor, ExecPolicy&& policy) const;
    template < std::copy_constructible Zero,
               std::copy_constructible Transform,
               std::copy_constructible Reduction,
               ExecutionPolicy_c       ExecPolicy >
    auto transfromReduce(Zero zero, Transform trans, Reduction reduction, ExecPolicy&& policy) const -> Zero
        requires BoundaryTransformReducible_c< Zero, Transform, Reduction, orders... >;

    [[nodiscard]] auto size() const -> size_t { return m_boundary_elements.size(); }

private:
    detail::boundary_element_view_variant_array_t< orders... > m_boundary_elements;
};

template < el_o_t... orders >
template < BoundaryViewVisitor_c< orders... > Visitor, ExecutionPolicy_c ExecPolicy >
void BoundaryView< orders... >::visit(Visitor&& visitor, ExecPolicy&& policy) const
{
    std::for_each(std::forward< ExecPolicy >(policy),
                  m_boundary_elements.begin(),
                  m_boundary_elements.end(),
                  [&visitor](const auto& boundary_el) { std::visit(visitor, boundary_el); });
}

template < el_o_t... orders >
template < std::copy_constructible Zero,
           std::copy_constructible Transform,
           std::copy_constructible Reduction,
           ExecutionPolicy_c       ExecPolicy >
auto BoundaryView< orders... >::transfromReduce(Zero         zero,
                                                Transform    trans,
                                                Reduction    reduction,
                                                ExecPolicy&& policy) const -> Zero
    requires BoundaryTransformReducible_c< Zero, Transform, Reduction, orders... >
{
    const auto transform_variant = [&](const auto& var) {
        return std::visit< Zero >(trans, var);
    };
    return std::transform_reduce(std::forward< ExecPolicy >(policy),
                                 m_boundary_elements.begin(),
                                 m_boundary_elements.end(),
                                 zero,
                                 reduction,
                                 transform_variant);
}
} // namespace lstr::mesh
#endif // L3STER_MESH_BOUNDARYVIEW_HPP
