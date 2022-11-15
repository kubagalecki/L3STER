#ifndef L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP
#define L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"

namespace lstr
{
template < ElementTypes ET, el_o_t EO, size_t n_points >
struct ReferenceBasisAtPoints
{
private:
    using basis_at_qp_t = EigenRowMajorMatrix< val_t, n_points, Element< ET, EO >::n_nodes >;
    using basis_ders_t  = std::array< basis_at_qp_t, Element< ET, EO >::native_dim >;

public:
    basis_at_qp_t values;
    basis_ders_t  derivatives;
};

namespace detail
{
template < BasisTypes                                                    BT,
           ElementTypes                                                  ET,
           el_o_t                                                        EO,
           std::convertible_to< Point< Element< ET, EO >::native_dim > > Point_t,
           size_t                                                        n_points >
auto evalRefBasisAtPoints(const std::array< Point_t, n_points >& points) -> ReferenceBasisAtPoints< ET, EO, n_points >
{
    constexpr auto                                  n_nodes    = Element< ET, EO >::n_nodes;
    constexpr auto                                  native_dim = Element< ET, EO >::native_dim;
    EigenRowMajorMatrix< val_t, n_points, n_nodes > basis_vals;
    std::array< decltype(basis_vals), native_dim >  basis_ders;
    for (size_t point_ind = 0; Point< native_dim > point : points)
    {
        basis_vals(point_ind, Eigen::all) = computeRefBasis< ET, EO, BT >(point);
        const auto ders_at_point          = computeRefBasisDers< ET, EO, BT >(point);
        for (size_t der_ind = 0; der_ind < native_dim; ++der_ind)
            basis_ders[der_ind](point_ind, Eigen::all) = ders_at_point(der_ind, Eigen::all);
        ++point_ind;
    }
    return {basis_vals, basis_ders};
}
} // namespace detail
} // namespace lstr
#endif // L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP
