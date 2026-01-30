#ifndef L3STER_MAPPING_JACOBIMAT_HPP
#define L3STER_MAPPING_JACOBIMAT_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Meta.hpp"

namespace lstr::map
{
// Jacobi matrix in native dimension: assumes y=0, z=0 for 1D, z=0 for 2D
template < mesh::ElementType ET >
using JacobiMat = Eigen::Matrix< val_t, mesh::Element< ET, 1 >::native_dim, mesh::Element< ET, 1 >::native_dim >;

// Note: the generator returned from this function cannot outlive the element data passed as its argument
template < mesh::ElementType T, el_o_t O >
auto getNatJacobiMatGenerator(const mesh::ElementData< T, O >& element_data)
{
    using traits       = mesh::ElementTraits< mesh::Element< T, O > >;
    constexpr auto GT  = traits::geom_type;
    constexpr auto GO  = traits::geom_order;
    constexpr auto GBT = basis::BasisType::Lagrange;
    return [&](const Point< traits::native_dim >& point) -> JacobiMat< T > {
        const auto shape_ders = basis::computeReferenceBasisDerivatives< GT, GO, GBT >(point);
        const auto vert_mat   = element_data.getEigenMap();
        return shape_ders * vert_mat.template topRows< traits::native_dim >().transpose();
    };
}
} // namespace lstr::map
#endif // L3STER_MAPPING_JACOBIMAT_HPP
