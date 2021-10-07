#ifndef L3STER_MAPPING_COMPUTEBASISDERIVATIVE_HPP
#define L3STER_MAPPING_COMPUTEBASISDERIVATIVE_HPP

#include "JacobiMat.hpp"

namespace lstr
{
template < el_locind_t I, BasisTypes BT, ElementTypes T, el_o_t O >
auto computePhysBasisDers(const Element< T, O >&                                       element,
                          const Point< ElementTraits< Element< T, O > >::native_dim >& point)
{
    constexpr auto nat_dim = ElementTraits< Element< T, O > >::native_dim;
    using vector_t         = Eigen::Matrix< val_t, nat_dim, 1 >;
    const auto jacobi_mat  = getNatJacobiMatGenerator(element)(point);
    vector_t   ref_ders;
    forConstexpr(
        [&]< dim_t DER_DIM >(std::integral_constant< dim_t, DER_DIM >) {
            ref_ders[DER_DIM] = ReferenceBasisFunction< T, O, I, BT, detail::derivativeByIndex(DER_DIM) >{}(point);
        },
        std::make_integer_sequence< dim_t, nat_dim >{});
    return vector_t{jacobi_mat.inverse() * ref_ders};
}

template < ElementTypes T, el_o_t O >
auto computePhysBasisDers(
    const Eigen::Matrix< val_t,
                         ElementTraits< Element< T, O > >::native_dim,
                         ElementTraits< Element< T, O > >::native_dim >&                                  jacobi_mat,
    const Eigen::Matrix< val_t, ElementTraits< Element< T, O > >::native_dim, Element< T, O >::n_nodes >& ref_ders)
{
    using ret_t = Eigen::Matrix< val_t, ElementTraits< Element< T, O > >::native_dim, Element< T, O >::n_nodes >;
    return ret_t{jacobi_mat.inverse() * ref_ders};
}
} // namespace lstr
#endif // L3STER_MAPPING_COMPUTEBASISDERIVATIVE_HPP
