#ifndef L3STER_ASSEMBLY_COMPUTEPHYSBASISDERSATPOINTS_HPP
#define L3STER_ASSEMBLY_COMPUTEPHYSBASISDERSATPOINTS_HPP

#include "l3ster/basisfun/ReferenceBasisAtQuadrature.hpp"
#include "l3ster/mapping/ComputeJacobiansAtPoints.hpp"
#include "l3ster/mapping/ComputePhysBasisDer.hpp"

namespace lstr
{
template < int native_dim, int n_bases, size_t n_points >
auto computePhysBasisDersAtPoints(
    const std::array< EigenRowMajorMatrix< val_t, native_dim, n_bases >, n_points >& ref_ders_at_points,
    const std::array< detail::jacobian_t< native_dim >, n_points >&                  jacobians_at_points)
{
    std::array< EigenRowMajorMatrix< val_t, native_dim, n_bases >, n_points > retval;
    std::ranges::transform(jacobians_at_points,
                           ref_ders_at_points,
                           begin(retval),
                           [](const auto& jac, const auto& ref_ders) { return computePhysBasisDers(jac, ref_ders); });
    return retval;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_COMPUTEPHYSBASISDERSATPOINTS_HPP
