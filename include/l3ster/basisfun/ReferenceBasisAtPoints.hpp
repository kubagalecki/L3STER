#ifndef L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP
#define L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"

namespace lstr
{
template < ElementTypes ET, el_o_t EO, size_t n_points >
struct ReferenceBasisAtPoints
{
    static constexpr auto n_bases = Element< ET, EO >::n_nodes;
    static constexpr auto dim     = Element< ET, EO >::native_dim;
    using basis_vals_t            = std::array< Eigen::Vector< val_t, n_bases >, n_points >;
    using basis_ders_t            = std::array< EigenRowMajorMatrix< val_t, dim, n_bases >, n_points >;

    basis_vals_t values;
    basis_ders_t derivatives;
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
    auto basis_vals = std::make_unique< typename ReferenceBasisAtPoints< ET, EO, n_points >::basis_vals_t >();
    auto basis_ders = std::make_unique< typename ReferenceBasisAtPoints< ET, EO, n_points >::basis_ders_t >();
    std::ranges::transform(points, begin(*basis_vals), [](auto pt) { return computeRefBasis< ET, EO, BT >(pt); });
    std::ranges::transform(points, begin(*basis_ders), [](auto pt) { return computeRefBasisDers< ET, EO, BT >(pt); });
    return {*basis_vals, *basis_ders};
}
} // namespace detail
} // namespace lstr
#endif // L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP
