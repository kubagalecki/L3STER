#ifndef L3STER_MESH_BOUNDARYVIEW_HPP
#define L3STER_MESH_BOUNDARYVIEW_HPP

#include "l3ster/mesh/BoundaryElementView.hpp"
#include "l3ster/util/UniVector.hpp"

namespace lstr::mesh
{
template < el_o_t... orders >
struct BoundaryView
{
    static_assert(sizeof...(orders) > 0);

    using bev_univec_t =
        parametrize_type_over_element_types_and_orders_t< util::UniVector, BoundaryElementView, orders... >;

    bev_univec_t element_views;
};
} // namespace lstr::mesh
#endif // L3STER_MESH_BOUNDARYVIEW_HPP
