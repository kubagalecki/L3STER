#ifndef L3STER_MAPPING_BOUNDARYINTEGRALJACOBIAN_HPP
#define L3STER_MAPPING_BOUNDARYINTEGRALJACOBIAN_HPP

#include "l3ster/mapping/JacobiMat.hpp"
#include "l3ster/mapping/ReferenceBoundaryToSideMapping.hpp"

namespace lstr::map
{
template < mesh::ElementType ET >
val_t computeBoundaryIntegralJacobian(el_side_t side, const JacobiMat< ET >& jacobi_mat)
{
    constexpr auto native_dim = mesh::ElementTraits< mesh::Element< ET, 1 > >::native_dim;
    constexpr auto GT         = mesh::ElementTraits< mesh::Element< ET, 1 > >::geom_type;
    if constexpr (native_dim == 1)
        return 0.;
    else if constexpr (native_dim == 2)
    {
        const auto& [rot_mat, _] = getReferenceBoundaryToSideMapping< GT >(side);
        return (jacobi_mat.transpose() * rot_mat.col(0)).norm();
    }
    else if constexpr (native_dim == 3)
    {
        const auto& [rot_mat, _] = getReferenceBoundaryToSideMapping< GT >(side);
        const auto d_shape_d_ref = (jacobi_mat.transpose() * rot_mat.template leftCols< 2 >()).eval();
        return d_shape_d_ref.col(0).cross(d_shape_d_ref.col(1)).norm();
    }
    else
        static_assert(util::always_false< ET >, "Only dimensions 1, 2, and 3 are currently supported");
}
} // namespace lstr::map
#endif // L3STER_MAPPING_BOUNDARYINTEGRALJACOBIAN_HPP
