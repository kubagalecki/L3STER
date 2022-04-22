#ifndef L3STER_ASSEMBLY_COMPUTEPHYSBASISDERSATQPOINTS_HPP
#define L3STER_ASSEMBLY_COMPUTEPHYSBASISDERSATQPOINTS_HPP

#include "l3ster/basisfun/ReferenceBasisAtQuadrature.hpp"
#include "l3ster/mapping/ComputeJacobiansAtQpoints.hpp"
#include "l3ster/mapping/ComputePhysBasisDer.hpp"

namespace lstr
{
template < int native_dim, int n_bases, int QL >
auto computePhysBasisDersAtQpoints(
    const std::array< Eigen::Matrix< val_t, QL, n_bases, Eigen::RowMajor >, size_t(native_dim) >& ref_ders_at_qp,
    const std::array< detail::jacobian_t< native_dim >, size_t(QL) >&                             jacobians_at_qp)
{
    std::array< Eigen::Matrix< val_t, QL, n_bases, Eigen::RowMajor >, native_dim > retval;
    for (size_t qp_ind = 0; qp_ind < QL; ++qp_ind)
    {
        const auto ref_ders = [&] {
            Eigen::Matrix< val_t, native_dim, n_bases, Eigen::RowMajor > ders_at_qp;
            for (int dim_ind = 0; dim_ind < native_dim; ++dim_ind)
                for (size_t bas_ind = 0; bas_ind < n_bases; ++bas_ind)
                    ders_at_qp(dim_ind, bas_ind) = ref_ders_at_qp[dim_ind](qp_ind, bas_ind);
            return ders_at_qp;
        }();
        const auto phys_ders = computePhysBasisDers(jacobians_at_qp[qp_ind], ref_ders);
        for (int dim_ind = 0; dim_ind < native_dim; ++dim_ind)
            for (size_t bas_ind = 0; bas_ind < n_bases; ++bas_ind)
                retval[dim_ind](qp_ind, bas_ind) = phys_ders(dim_ind, bas_ind);
    }
    return retval;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_COMPUTEPHYSBASISDERSATQPOINTS_HPP
