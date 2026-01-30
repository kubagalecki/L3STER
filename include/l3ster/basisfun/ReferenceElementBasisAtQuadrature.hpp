#ifndef L3STER_BASISFUN_REFERENCEELEMENTBASISATQUADRATURE_HPP
#define L3STER_BASISFUN_REFERENCEELEMENTBASISATQUADRATURE_HPP

#include "l3ster/basisfun/ReferenceBasisAtQuadrature.hpp"
#include "l3ster/basisfun/ReferenceBasisFunction.hpp"
#include "l3ster/mapping/ReferenceBoundaryToSideMapping.hpp"

namespace lstr::basis
{
template < BasisType BT, mesh::ElementType ET, el_o_t EO, quad::QuadratureType QT, q_o_t QO >
const auto& getReferenceBasisAtDomainQuadrature()
{
    static const auto value = std::invoke([] {
        constexpr auto GBT        = BasisType::Lagrange;
        constexpr auto GT         = mesh::ElementTraits< mesh::Element< ET, EO > >::geom_type;
        constexpr auto GO         = mesh::ElementTraits< mesh::Element< ET, EO > >::geom_order;
        const auto     quadrature = quad::getQuadrature< QT, QO, ET >();
        return ReferenceBasisAtQuadrature< ET, EO, quadrature.size >{
            .quadrature = quadrature,
            .basis      = evalRefBasisAtPoints< BT, ET, EO >(quadrature.points),
            .geom_basis = evalRefBasisAtPoints< GBT, GT, GO >(quadrature.points)};
    });
    return value;
}

namespace detail
{
template < q_l_t QL, dim_t QD >
auto getReferenceBoundaryQpCoords(const quad::Quadrature< QL, QD >& ref_qp)
{
    using retval_t = Eigen::Matrix< val_t, QD + 1, QL >;
    retval_t retval;
    for (int c = 0; c < static_cast< int >(QL); ++c)
    {
        int r = 0;
        for (; r < static_cast< int >(QD); ++r)
            retval(r, c) = ref_qp.points[c][r];
        retval(r, c) = 0.;
    }
    return retval;
}

template < int R, int C >
void translate(Eigen::Matrix< val_t, R, C >& mat, const Eigen::Vector< val_t, R >& t)
{
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r)
            mat(r, c) += t[r];
}

template < int R, int C >
auto makeQuadratureFromCoordsMat(const Eigen::Matrix< val_t, R, C >& coords, const auto& weights)
{
    auto points = typename quad::Quadrature< C, R >::q_points_t{};
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r)
            points[c][r] = coords(r, c);
    return quad::Quadrature< C, R >{points, weights};
}
} // namespace detail

template < BasisType BT, mesh::ElementType ET, el_o_t EO, quad::QuadratureType QT, q_o_t QO >
const auto& getReferenceBasisAtBoundaryQuadrature(el_side_t el_side)
{
    // Assumption: quadratures constructed for integration over all sides of the element have the same number of
    // points. If at some point in the future, e.g., pyramids are supported, this will not be true. In that event,
    // quadratures for the individual sides will need to be stored in a tuple, and the result will need to be a
    // variant. Until then, an array + known return value is much simpler.
    using traits                   = mesh::ElementTraits< mesh::Element< ET, EO > >;
    static const auto lookup_table = std::invoke([] {
        const auto [ref_quad_coords, weights] = std::invoke([] {
            if constexpr (traits::native_dim > 1)
            {
                using enum mesh::ElementType;
                constexpr auto boundary_type = traits::geom_type == Hex ? Quad : Line;
                const auto     ref_quad      = quad::getQuadrature< QT, QO, boundary_type >();
                return std::make_pair(detail::getReferenceBoundaryQpCoords(ref_quad), ref_quad.weights);
            }
            else
            {
                const auto coords = Eigen::Vector< val_t, 1 >::Zero().eval();
                return std::make_pair(coords, std::array< val_t, 1 >{1.});
            }
        });
        constexpr auto GBT                    = BasisType::Lagrange;
        constexpr auto GT                     = traits::geom_type;
        constexpr auto GO                     = traits::geom_order;
        constexpr auto n_sides                = traits::n_sides;
        constexpr auto boundary_quad_size     = std::remove_const_t< decltype(ref_quad_coords) >::ColsAtCompileTime;
        using ref_basis_t                     = ReferenceBasisAtQuadrature< ET, EO, boundary_quad_size >;
        auto retval                           = std::array< ref_basis_t, n_sides >{};
        for (auto&& [side, side_quadrature] : retval | std::views::enumerate)
        {
            const auto [rot, trans] = map::getReferenceBoundaryToSideMapping< GT >(static_cast< el_side_t >(side));
            std::remove_const_t< decltype(ref_quad_coords) > side_quad_coords = rot * ref_quad_coords;
            detail::translate(side_quad_coords, trans);
            const auto quadrature = detail::makeQuadratureFromCoordsMat(side_quad_coords, weights);
            side_quadrature       = ReferenceBasisAtQuadrature< ET, EO, quadrature.size >{
                      .quadrature = quadrature,
                      .basis      = evalRefBasisAtPoints< BT, ET, EO >(quadrature.points),
                      .geom_basis = evalRefBasisAtPoints< GBT, GT, GO >(quadrature.points)};
        }
        return retval;
    });
    return lookup_table[el_side];
}
} // namespace lstr::basis
#endif // L3STER_BASISFUN_REFERENCEELEMENTBASISATQUADRATURE_HPP
