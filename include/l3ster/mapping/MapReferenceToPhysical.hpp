#ifndef L3STER_MAPPING_MAPREFERENCETOPHYSICAL_HPP
#define L3STER_MAPPING_MAPREFERENCETOPHYSICAL_HPP

#include "l3ster/basisfun/BasisAtPoints.hpp"
#include "l3ster/mapping/BoundaryIntegralJacobian.hpp"
#include "l3ster/mapping/BoundaryNormal.hpp"
#include "l3ster/util/Caliper.hpp"

namespace lstr::map
{
template < size_t NG, size_t NP >
auto mapToPhysicalSpace(const basis::TabulatedBasisValues< NG, NP >& geom_basis,
                        std::span< const Point< 3 >, NG >            vertices) -> std::array< Point< 3 >, NP >
{
    static_assert(NG > 0 && NP > 0);
    L3STER_PROFILE_FUNCTION;
    using vert_map_t         = Eigen::Map< const Eigen::Matrix< val_t, 3, NG > >;
    using point_map_t        = Eigen::Map< Eigen::Matrix< val_t, 3, NP > >;
    const auto vert_map      = vert_map_t{vertices.front().coords.data()};
    const auto geo_basis_map = geom_basis.getMap();

    std::array< Point< 3 >, NP > retval;
    auto                         point_map = point_map_t{retval.front().coords.data()};

    point_map.transpose() = geo_basis_map * vert_map.transpose();
    return retval;
}

template < size_t NA, size_t NP, dim_t D >
struct TabulatedDomainMapping
{
    template < size_t NG >
    TabulatedDomainMapping(const basis::TabulatedBasisAtPointsView< NA, NG, NP, D >& basis,
                           std::span< const Point< 3 >, NG >                         vertices)
    {
        L3STER_PROFILE_FUNCTION;
        auto jacobi_mats = computeJacobiMats(basis.geom_basis->derivatives, vertices);
        jacobians        = computeJacobians(jacobi_mats);
        invertJacobiMats(jacobi_mats, std::span{std::as_const(jacobians)});
        basis_values         = &basis.approx_basis->values;
        physical_derivatives = computePhysicalDerivatives(basis.approx_basis->derivatives, jacobi_mats);
        points               = mapToPhysicalSpace(basis.geom_basis->values, vertices);
    }

    std::array< val_t, NP >                       jacobians;
    const basis::TabulatedBasisValues< NA, NP >*  basis_values;
    basis::TabulatedBasisDerivatives< NA, NP, D > physical_derivatives;
    std::array< Point< 3 >, NP >                  points;
};

template < size_t NA, size_t NP, dim_t D >
struct TabulatedBoundaryMapping
{
    template < size_t NG, mesh::ElementType GT >
    TabulatedBoundaryMapping(const basis::TabulatedBasisAtPointsView< NA, NG, NP, D >& basis,
                             std::span< const Point< 3 >, NG >                         vertices,
                             util::ConstexprValue< GT >,
                             el_side_t side)
    {
        L3STER_PROFILE_FUNCTION;
        auto jacobi_mats = computeJacobiMats(basis.geom_basis->derivatives, vertices);
        for (size_t p = 0; p != NP; ++p)
            normals[p] = computeBoundaryNormal< GT >(side, jacobi_mats.get(p));
        const auto volume_jacobians = computeJacobians(jacobi_mats);
        jacobians                   = computeBoundaryIntegralJacobians< GT >(side, jacobi_mats);
        invertJacobiMats(jacobi_mats, std::span{volume_jacobians});
        basis_values         = &basis.approx_basis->values;
        physical_derivatives = computePhysicalDerivatives(basis.approx_basis->derivatives, jacobi_mats);
        points               = mapToPhysicalSpace(basis.geom_basis->values, vertices);
    }

    std::array< val_t, NP >                       jacobians;
    const basis::TabulatedBasisValues< NA, NP >*  basis_values;
    basis::TabulatedBasisDerivatives< NA, NP, D > physical_derivatives;
    std::array< Point< 3 >, NP >                  points;
    std::array< Eigen::Vector< val_t, D >, NP >   normals;
};

template < size_t NP, size_t NF, dim_t D >
class FieldValuesAtPoints
{
public:
    using vals_t = std::array< val_t, NF >;
    using ders_t = std::array< vals_t, D >;

    FieldValuesAtPoints()
        requires(NF == 0)
    = default;
    template < size_t NA >
    FieldValuesAtPoints(const basis::TabulatedBasisValues< NA, NP >&         basis_vals,
                        const basis::TabulatedBasisDerivatives< NA, NP, D >& basis_phys_ders,
                        const Eigen::Matrix< val_t, int{NA}, int{NF} >&      node_vals)
    {
        L3STER_PROFILE_FUNCTION;
        if constexpr (NF > 0) // Eigen matrix assignment for size=0 doesn't compile
        {
            m_fields.getValuesMap() = basis_vals.getMap() * node_vals;
            for (dim_t d = 0; d != D; ++d)
                m_fields.getDerivativesMap(d) = basis_phys_ders.getMap(d) * node_vals;
        }
    }

    auto get(size_t point) const noexcept -> std::pair< vals_t, ders_t >
    {
        std::pair< vals_t, ders_t > retval;
        auto& [vals, ders] = retval;
        for (size_t f = 0; f != NF; ++f)
        {
            vals[f] = m_fields.getValuesMap()(point, f);
            for (dim_t d = 0; d != D; ++d)
                ders[d][f] = m_fields.getDerivativesMap(d)(point, f);
        }
        return retval;
    }

private:
    basis::TabulatedBasis< NF, NP, D > m_fields;
};
template < size_t NA, size_t NP, int NF, dim_t D >
FieldValuesAtPoints(const basis::TabulatedBasisValues< NA, NP >&,
                    const basis::TabulatedBasisDerivatives< NA, NP, D >&,
                    const Eigen::Matrix< val_t, int{NA}, NF >&)
    -> FieldValuesAtPoints< NP, static_cast< size_t >(NF), D >;

template < mesh::ElementType ET, el_o_t EO >
auto getPhysicalNodeLocations(const mesh::Element< ET, EO >& element)
{
    L3STER_PROFILE_FUNCTION;
    using eltraits            = mesh::ElementTraits< mesh::Element< ET, EO > >;
    constexpr auto GBT        = basis::BasisType::Lagrange;
    const auto&    geom_basis = basis::getBasisAtNodes< GBT, eltraits::geom_type, EO, eltraits::geom_order >();
    const auto     verts      = std::span{element.data.vertices};
    return mapToPhysicalSpace(geom_basis.values, verts);
}

template < mesh::ElementType ET, el_o_t EO >
auto getPhysicalSideNodeLocations(const mesh::BoundaryElementView< ET, EO >& el_view)
{
    L3STER_PROFILE_FUNCTION;
    using eltraits            = mesh::ElementTraits< mesh::Element< ET, EO > >;
    constexpr auto GBT        = basis::BasisType::Lagrange;
    const auto     side       = el_view.getSide();
    const auto&    geom_basis = basis::getBasisAtSideNodes< GBT, eltraits::geom_type, EO, eltraits::geom_order >(side);
    const auto     verts      = std::span{el_view->data.vertices};
    return mapToPhysicalSpace(geom_basis.values, verts);
}
} // namespace lstr::map
#endif // L3STER_MAPPING_MAPREFERENCETOPHYSICAL_HPP
