#ifndef L3STER_MAPPING_BOUNDARYINTEGRALJACOBIAN_HPP
#define L3STER_MAPPING_BOUNDARYINTEGRALJACOBIAN_HPP

#include "l3ster/mapping/ReferenceBoundaryToSideMapping.hpp"
#include "l3ster/mesh/BoundaryElementView.hpp"

namespace lstr::map
{
template < ElementType ET, el_o_t EO >
val_t computeBoundaryIntegralJacobian(
    BoundaryElementView< ET, EO >                                                               el_view,
    const Eigen::Matrix< val_t, Element< ET, EO >::native_dim, Element< ET, EO >::native_dim >& jacobi_mat)
{
    if constexpr (Element< ET, EO >::native_dim == 1)
        return 0.;
    else
    {
        const auto& [rot_mat, _] = getReferenceBoundaryToSideMapping< ET >(el_view.getSide());
        if constexpr (Element< ET, EO >::native_dim == 2)
            return (jacobi_mat.transpose() * rot_mat.col(0)).norm();
        else if constexpr (Element< ET, EO >::native_dim == 3)
        {
            const Eigen::Matrix< val_t, 3, 2 > d_shape_d_ref =
                jacobi_mat.transpose() * rot_mat(Eigen::all, Eigen::seq(Eigen::fix< 0 >, Eigen::fix< 1 >));
            return d_shape_d_ref.col(0).cross(d_shape_d_ref.col(1)).norm();
        }
        else
            static_assert(ET != ET, "Only dimensions 1, 2, and 3 are currently supported");
    }
}
} // namespace lstr::map
#endif // L3STER_MAPPING_BOUNDARYINTEGRALJACOBIAN_HPP
