#ifndef L3STER_MESH_BOUNDARYVIEW_HPP
#define L3STER_MESH_BOUNDARYVIEW_HPP

#include "l3ster/mesh/BoundaryElementView.hpp"
#include "l3ster/util/ArrayOwner.hpp"
#include "l3ster/util/Ranges.hpp"
#include "l3ster/util/UniVector.hpp"

#include <utility>

namespace lstr::mesh
{
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

    using bev_univec_t =
        parametrize_type_over_element_types_and_orders_t< util::UniVector, BoundaryElementView, orders... >;

public:
    bev_univec_t element_views;
};
} // namespace lstr::mesh
#endif // L3STER_MESH_BOUNDARYVIEW_HPP
