#ifndef L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP
#define L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"
#include "l3ster/mesh/NodeLocation.hpp"

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

template < BasisType                                                           BT,
           mesh::ElementType                                                   ET,
           el_o_t                                                              EO,
           std::convertible_to< Point< mesh::Element< ET, EO >::native_dim > > Point_t,
           size_t                                                              n_points >
auto evalRefBasisAtPoints(const std::array< Point_t, n_points >& pts) -> ReferenceBasisAtPoints< ET, EO, n_points >
{
    using traits          = mesh::ElementTraits< mesh::Element< ET, EO > >;
    constexpr auto GT     = traits::geom_type;
    auto           retval = ReferenceBasisAtPoints< ET, EO, n_points >{};
    auto& [vals, ders]    = retval;
    std::ranges::transform(pts, vals.begin(), [](auto p) { return computeReferenceBases< GT, EO, BT >(p); });
    std::ranges::transform(pts, ders.begin(), [](auto p) { return computeReferenceBasisDerivatives< GT, EO, BT >(p); });
    return retval;
}

template < mesh::ElementType ET, el_o_t EO, BasisType BT = BasisType::Lagrange >
const auto& getBasisAtNodes()
{
    static const ReferenceBasisAtPoints< ET, EO, mesh::Element< ET, EO >::n_nodes > retval = std::invoke([] {
        const auto& reference_locations = mesh::getNodeLocations< ET, EO >();
        return evalRefBasisAtPoints< BT, ET, EO >(reference_locations);
    });
    return retval;
}
} // namespace lstr::basis
#endif // L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP
