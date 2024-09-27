#ifndef L3STER_MAPPING_BOUNDARYINTEGRALJACOBIAN_HPP
#define L3STER_MAPPING_BOUNDARYINTEGRALJACOBIAN_HPP

#include "l3ster/mapping/JacobiMat.hpp"
#include "l3ster/mapping/ReferenceBoundaryToSideMapping.hpp"

namespace lstr::map
{
template < mesh::ElementType ET, el_o_t EO >
val_t computeBoundaryIntegralJacobian(el_side_t side, const JacobiMat< ET, EO >& jacobi_mat)
{
    if constexpr (mesh::Element< ET, EO >::native_dim == 1)
        return 0.;
    else
    {
        const auto& [rot_mat, _] = getReferenceBoundaryToSideMapping< ET >(side);
        if constexpr (mesh::Element< ET, EO >::native_dim == 2)
            return (jacobi_mat.transpose() * rot_mat.col(0)).norm();
        else if constexpr (mesh::Element< ET, EO >::native_dim == 3)
        {
            const Eigen::Matrix< val_t, 3, 2 > d_shape_d_ref =
                jacobi_mat.transpose() * rot_mat.template leftCols< 2 >();
            return d_shape_d_ref.col(0).cross(d_shape_d_ref.col(1)).norm();
        }
        else
            static_assert(util::always_false< ET >, "Only dimensions 1, 2, and 3 are currently supported");
    }
}
} // namespace lstr::map
#endif // L3STER_MAPPING_BOUNDARYINTEGRALJACOBIAN_HPP
