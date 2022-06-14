#ifndef L3STER_BASISFUN_REFERENCEELEMENTBASISATQUADRATURE_HPP
#define L3STER_BASISFUN_REFERENCEELEMENTBASISATQUADRATURE_HPP

#include "l3ster/basisfun/ComputeRefBasisAtQpoints.hpp"
#include "l3ster/basisfun/ReferenceBasisAtQuadrature.hpp"

namespace lstr
{
template < BasisTypes BT, ElementTypes ET, el_o_t EO, QuadratureTypes QT, q_o_t QO >
const auto& getReferenceBasisAtDomainQuadrature()
{
    static const auto value = [] {
        const auto quadrature = getQuadrature< QT, QO, ET >();
        const auto basis_vals = computeRefBasisAtQpoints< BT, ET, EO >(quadrature);
        const auto basis_ders = computeRefBasisDersAtQpoints< BT, ET, EO >(quadrature);
        return ReferenceBasisAtQuadrature< ET, EO, quadrature.size, quadrature.dim >{
            .quadrature = quadrature, .basis_vals = basis_vals, .basis_ders = basis_ders};
    }();
    return value;
}

namespace detail
{
template < q_l_t QL, dim_t QD >
auto getReferenceBoundaryQpCoords(const Quadrature< QL, QD >& ref_qp)
{
    using retval_t = Eigen::Matrix< val_t, QD + 1, QL >;
    retval_t retval;
    for (int c = 0; c < static_cast< int >(QL); ++c)
    {
        int r = 0;
        for (; r < static_cast< int >(QD); ++r)
            retval(r, c) = ref_qp.getPoints()[c][r];
        retval(r, c) = 0.;
    }
    return retval;
}

template < int C >
auto rotate2D90(const Eigen::Matrix< val_t, 2, C >& mat) -> Eigen::Matrix< val_t, 2, C >
{
    Eigen::Matrix< val_t, 2, 2 > rot_mat;
    rot_mat(0, 0) = 0.;
    rot_mat(0, 1) = -1.;
    rot_mat(1, 0) = 1.;
    rot_mat(1, 1) = 0.;
    return rot_mat * mat;
}

template < int C >
auto rotate3D90(const Eigen::Matrix< val_t, 3, C >& mat, Space axis) -> Eigen::Matrix< val_t, 3, C >
{
    Eigen::Matrix< val_t, 3, 3 > rot_mat = Eigen::Matrix< val_t, 3, 3 >::Zero();
    switch (axis)
    {
    case Space::X:
        rot_mat(0, 0) = 1.;
        rot_mat(1, 2) = -1.;
        rot_mat(2, 1) = 1.;
        break;
    case Space::Y:
        rot_mat(1, 1) = 1.;
        rot_mat(0, 2) = 1.;
        rot_mat(2, 0) = -1.;
        break;
    case Space::Z: // not actually needed, the input in 3D is perpendicular to the Z axis
        rot_mat(2, 2) = 1.;
        rot_mat(0, 1) = -1.;
        rot_mat(1, 0) = 1.;
    }
    return rot_mat * mat;
}

template < int R, int C >
void translate(Eigen::Matrix< val_t, R, C >& mat, const Eigen::Matrix< val_t, R, 1 >& t)
{
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r)
            mat(r, c) += t[r];
}

template < int R, int C >
auto makeQuadratureFromCoordsMat(const Eigen::Matrix< val_t, R, C >& coords, const auto& weights)
{
    // Note: there's possibly a smarter way of doing this, since the layout of `points` and `coords` is the same, but
    // perf is irrelevant here
    typename Quadrature< C, R >::q_points_t points;
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r)
            points[c][r] = coords(r, c);
    return Quadrature< C, R >{points, weights};
}
} // namespace detail

template < BasisTypes BT, ElementTypes ET, el_o_t EO, QuadratureTypes QT, q_o_t QO >
const auto& getReferenceBasisAtBoundaryQuadrature(el_side_t side)
    requires(ET == ElementTypes::Quad or ET == ElementTypes::Hex)
{
    // Assumption: quadratures constructed for integration over all sides of the element have the same number of points.
    // If at some point in the future, e.g., pyramids are supported, this will not be true. In that event, quadratures
    // for the individual sides will need to be stored in a tuple, and the result will need to be a variant. Until then,
    // an array + known return value is much simpler.

    static const auto values = [] {
        constexpr auto boundary_type   = ET == ElementTypes::Hex ? ElementTypes::Quad : ElementTypes::Line;
        const auto     boundary_quad   = getQuadrature< QT, QO, boundary_type >();
        const auto     ref_quad_coords = detail::getReferenceBoundaryQpCoords(boundary_quad);
        using ref_basis_t = ReferenceBasisAtQuadrature< ET, EO, boundary_quad.size, boundary_quad.dim + 1 >;

        constexpr auto                        n_el_sides = ElementTraits< Element< ET, EO > >::n_sides;
        std::array< ref_basis_t, n_el_sides > quadrature_array;
        for (el_side_t side_ind = 0; side_ind < n_el_sides; ++side_ind)
        {
            std::remove_const_t< decltype(ref_quad_coords) > qp_coords;
            if constexpr (ET == ElementTypes::Quad)
            {
                switch (side_ind)
                {
                case 0:
                    qp_coords = ref_quad_coords;
                    detail::translate(qp_coords, Eigen::Vector2d(0., -1.));
                    break;
                case 1:
                    qp_coords = ref_quad_coords;
                    detail::translate(qp_coords, Eigen::Vector2d(0., 1.));
                    break;
                case 2:
                    qp_coords = detail::rotate2D90(ref_quad_coords);
                    detail::translate(qp_coords, Eigen::Vector2d(-1., 0.));
                    break;
                case 3:
                    qp_coords = detail::rotate2D90(ref_quad_coords);
                    detail::translate(qp_coords, Eigen::Vector2d(1., 0.));
                    break;
                }
            }
            else if constexpr (ET == ElementTypes::Hex)
            {
                switch (side_ind)
                {
                case 0:
                    qp_coords = detail::rotate3D90(ref_quad_coords, Space::Z);
                    detail::translate(qp_coords, Eigen::Vector3d(0., 0., -1.));
                    break;
                case 1:
                    qp_coords = detail::rotate3D90(ref_quad_coords, Space::Z);
                    detail::translate(qp_coords, Eigen::Vector3d(0., 0., 1.));
                    break;
                case 2:
                    qp_coords = detail::rotate3D90(ref_quad_coords, Space::X);
                    detail::translate(qp_coords, Eigen::Vector3d(0., -1., 0.));
                    break;
                case 3:
                    qp_coords = detail::rotate3D90(ref_quad_coords, Space::X);
                    detail::translate(qp_coords, Eigen::Vector3d(0., 1., 0.));
                    break;
                case 4:
                    qp_coords = detail::rotate3D90(ref_quad_coords, Space::Y);
                    detail::translate(qp_coords, Eigen::Vector3d(-1., 0., 0.));
                    break;
                case 5:
                    qp_coords = detail::rotate3D90(ref_quad_coords, Space::Y);
                    detail::translate(qp_coords, Eigen::Vector3d(1., 0., 0.));
                    break;
                }
            }
            const auto quadrature      = detail::makeQuadratureFromCoordsMat(qp_coords, boundary_quad.getWeights());
            const auto basis_vals      = computeRefBasisAtQpoints< BT, ET, EO >(quadrature);
            const auto basis_ders      = computeRefBasisDersAtQpoints< BT, ET, EO >(quadrature);
            quadrature_array[side_ind] = ReferenceBasisAtQuadrature< ET, EO, quadrature.size, quadrature.dim >{
                .quadrature = quadrature, .basis_vals = basis_vals, .basis_ders = basis_ders};
        }
        return quadrature_array;
    }();
    return values[side];
}
} // namespace lstr
#endif // L3STER_BASISFUN_REFERENCEELEMENTBASISATQUADRATURE_HPP
