#ifndef L3STER_ASSEMBLY_COMPUTEPHYSBASISDERSATPOINTS_HPP
#define L3STER_ASSEMBLY_COMPUTEPHYSBASISDERSATPOINTS_HPP

#include "l3ster/basisfun/ReferenceBasisAtQuadrature.hpp"
#include "l3ster/mapping/ComputeJacobiansAtPoints.hpp"
#include "l3ster/mapping/ComputePhysBasisDer.hpp"

namespace lstr
{
template < int native_dim, int n_bases, int n_points >
auto computePhysBasisDersAtPoints(
    const std::array< EigenRowMajorMatrix< val_t, n_points, n_bases >, size_t(native_dim) >& ref_ders_at_points,
    const std::array< detail::jacobian_t< native_dim >, size_t(n_points) >&                  jacobians_at_points)
{
    std::array< EigenRowMajorMatrix< val_t, n_points, n_bases >, native_dim > retval;
    for (size_t qp_ind = 0; qp_ind < static_cast< size_t >(n_points); ++qp_ind)
    {
        const auto ref_ders  = std::invoke([&] {
            EigenRowMajorMatrix< val_t, native_dim, n_bases > ders_at_qp;
            for (int dim_ind = 0; dim_ind < native_dim; ++dim_ind)
                for (size_t bas_ind = 0; bas_ind < n_bases; ++bas_ind)
                    ders_at_qp(dim_ind, bas_ind) = ref_ders_at_points[dim_ind](qp_ind, bas_ind);
            return ders_at_qp;
        });
        const auto phys_ders = computePhysBasisDers(jacobians_at_points[qp_ind], ref_ders);
        for (size_t dim_ind = 0; dim_ind < static_cast< size_t >(native_dim); ++dim_ind)
            for (size_t bas_ind = 0; bas_ind < static_cast< size_t >(n_bases); ++bas_ind)
                retval[dim_ind](qp_ind, bas_ind) = phys_ders(dim_ind, bas_ind);
    }
    return retval;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_COMPUTEPHYSBASISDERSATPOINTS_HPP
