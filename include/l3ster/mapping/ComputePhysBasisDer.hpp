#ifndef L3STER_MAPPING_COMPUTEPHYSBASISDER_HPP
#define L3STER_MAPPING_COMPUTEPHYSBASISDER_HPP

#include "l3ster/defs/Typedefs.h"
#include "l3ster/util/IncludeEigen.hpp"

namespace lstr
{
template < int native_dim, int n_bases >
auto computePhysBasisDers(const Eigen::Matrix< val_t, native_dim, native_dim >&    jacobi_mat,
                          const EigenRowMajorMatrix< val_t, native_dim, n_bases >& ref_ders)
{
    return Eigen::Matrix< val_t, native_dim, n_bases >{jacobi_mat.inverse() * ref_ders};
}
} // namespace lstr
#endif // L3STER_MAPPING_COMPUTEPHYSBASISDER_HPP
