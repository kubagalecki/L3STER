#ifndef L3STER_MAPPING_COMPUTEPHYSBASISDER_HPP
#define L3STER_MAPPING_COMPUTEPHYSBASISDER_HPP

#include "l3ster/common/Typedefs.h"
#include "l3ster/util/EigenUtils.hpp"

namespace lstr::map
{
template < int native_dim, int n_bases >
auto computePhysBasisDers(const Eigen::Matrix< val_t, native_dim, native_dim >&            jacobi_mat,
                          const util::eigen::RowMajorMatrix< val_t, native_dim, n_bases >& ref_ders)
    -> Eigen::Matrix< val_t, native_dim, n_bases >
{
    return Eigen::Matrix< val_t, native_dim, n_bases >{jacobi_mat.inverse() * ref_ders};
}
} // namespace lstr::map
#endif // L3STER_MAPPING_COMPUTEPHYSBASISDER_HPP
