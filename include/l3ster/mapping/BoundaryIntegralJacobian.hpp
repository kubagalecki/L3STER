#ifndef L3STER_MAPPING_BOUNDARYINTEGRALJACOBIAN_HPP
#define L3STER_MAPPING_BOUNDARYINTEGRALJACOBIAN_HPP

#include "l3ster/mapping/ComputePhysicalDerivatives.hpp"
#include "l3ster/mapping/ReferenceBoundaryToSideMapping.hpp"

namespace lstr::map
{
template < mesh::ElementType ET, size_t NP >
auto computeBoundaryIntegralJacobians(el_side_t                                                   side,
                                      const JacobiMats< NP, mesh::Element< ET, 1 >::native_dim >& jacobi_mats)
    -> std::array< val_t, NP >
    requires(mesh::isGeomType(ET))
{
    constexpr auto native_dim = mesh::ElementTraits< mesh::Element< ET, 1 > >::native_dim;
    const auto& [rot_mat, _]  = getReferenceBoundaryToSideMapping< ET >(side);
    auto retval               = std::array< val_t, NP >{};
    if constexpr (native_dim == 2)
    {
        for (size_t p = 0; p != NP; ++p)
            retval[p] = (jacobi_mats.get(p).transpose() * rot_mat.col(0)).norm();
    }
    else if constexpr (native_dim == 3)
    {
        for (size_t p = 0; p != NP; ++p)
        {
            const auto jm            = jacobi_mats.get(p);
            const auto d_shape_d_ref = (jm.transpose() * rot_mat.template leftCols< 2 >()).eval();
            retval[p]                = d_shape_d_ref.col(0).cross(d_shape_d_ref.col(1)).norm();
        }
    }
    return retval;
}
} // namespace lstr::map
#endif // L3STER_MAPPING_BOUNDARYINTEGRALJACOBIAN_HPP
