#ifndef L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP
#define L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"

namespace lstr::basis
{
template < mesh::ElementType ET, el_o_t EO, size_t n_points >
struct ReferenceBasisAtPoints
{
    static constexpr auto n_bases   = mesh::Element< ET, EO >::n_nodes;
    static constexpr auto n_qpoints = n_points;
    static constexpr auto dim       = mesh::Element< ET, EO >::native_dim;
    using basis_vals_t              = std::array< Eigen::Vector< val_t, n_bases >, n_points >;
    using basis_ders_t              = std::array< util::eigen::RowMajorMatrix< val_t, dim, n_bases >, n_points >;

    basis_vals_t values;
    basis_ders_t derivatives;
};

namespace detail
{
template < BasisType                                                           BT,
           mesh::ElementType                                                   ET,
           el_o_t                                                              EO,
           std::convertible_to< Point< mesh::Element< ET, EO >::native_dim > > Point_t,
           size_t                                                              n_points >
auto evalRefBasisAtPoints(const std::array< Point_t, n_points >& pts) -> ReferenceBasisAtPoints< ET, EO, n_points >
{
    auto retval        = ReferenceBasisAtPoints< ET, EO, n_points >{};
    auto& [vals, ders] = retval;
    std::ranges::transform(pts, vals.begin(), [](auto p) { return computeReferenceBases< ET, EO, BT >(p); });
    std::ranges::transform(pts, ders.begin(), [](auto p) { return computeReferenceBasisDerivatives< ET, EO, BT >(p); });
    return retval;
}
} // namespace detail
} // namespace lstr::basis
#endif // L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP
