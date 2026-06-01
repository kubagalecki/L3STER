#ifndef L3STER_BASIS_BASISATQUADRATURE_HPP
#define L3STER_BASIS_BASISATQUADRATURE_HPP

#include "l3ster/basisfun/TabulatedBasis.hpp"
#include "l3ster/mapping/ReferenceBoundaryToSideMapping.hpp"
#include "l3ster/mesh/NodeLocation.hpp"
#include "l3ster/quad/GenerateQuadrature.hpp"

namespace lstr::basis
{
namespace detail
{
template < size_t QD >
auto liftDim(std::array< val_t, QD > point)
{
    auto retval = std::array< val_t, QD + 1 >{};
    std::ranges::copy(point, retval.begin());
    return retval;
}

template < size_t QD >
auto mapToBoundary(const std::array< val_t, QD >&                  point,
                   const Eigen::Matrix< val_t, int{QD}, int{QD} >& rot_mat,
                   const Eigen::Vector< val_t, int{QD} >&          translation) -> Point< QD >
{
    const auto     point_vec  = Eigen::Map< const Eigen::Vector< val_t, QD > >{point.data()};
    const auto     rotated    = (rot_mat * point_vec).eval();
    const auto     translated = (rotated + translation).eval();
    constexpr auto inds       = util::makeIotaArray< int, QD >();
    return {util::elwise(inds, [&translated](int i) { return translated[i]; })};
}
} // namespace detail

template < BasisType BT, mesh::ElementType ET, el_o_t EO, quad::QuadratureType QT, q_o_t QO >
const auto& getBasisInDomain()
    requires(mesh::isGeomType(ET))
{
    static const auto value = std::invoke([] {
        const auto& quadrature = quad::getQuadrature< QT, QO, ET >();
        const auto  points     = util::elwise(quadrature.points, [](auto arr) { return Point{arr}; });
        return tabulateBasis< BT, ET, EO >(std::span{points});
    });
    return value;
}

template < BasisType BT, mesh::ElementType ET, el_o_t EO, quad::QuadratureType QT, q_o_t QO >
const auto& getBasisAtBoundary(el_side_t el_side)
    requires(mesh::isGeomType(ET))
{
    // Assumption: quadratures constructed for integration over all sides of the element have the same number of
    // points. If at some point in the future, e.g., pyramids are supported, this will not be true. In that event,
    // quadratures for the individual sides will need to be stored in a tuple, and the result will need to be a
    // variant. Until then, an array + known return value is much simpler.
    using eltraits_t               = mesh::ElementTraits< mesh::Element< ET, EO > >;
    static const auto lookup_table = std::invoke([] {
        const auto ref_qps        = std::invoke([] {
            if constexpr (eltraits_t::native_dim > 1)
            {
                using enum mesh::ElementType;
                constexpr auto boundary_type = eltraits_t::geom_type == Hex ? Quad : Line;
                const auto& [bq, w]          = quad::getQuadrature< QT, QO, boundary_type >();
                return util::elwise(bq, [](auto p) { return detail::liftDim(p); });
            }
            else
                return std::array< std::array< val_t, 1 >, 1 >{};
        });
        const auto make_side_quad = [&](el_side_t side) {
            const auto [rot, trans] = map::getReferenceBoundaryToSideMapping< ET >(side);
            const auto qps = util::elwise(ref_qps, [&](auto p) { return detail::mapToBoundary(p, rot, trans); });
            return tabulateBasis< BT, ET, EO >(std::span{qps});
        };
        constexpr auto side_inds = util::makeIotaArray< el_side_t, eltraits_t::n_sides >();
        return util::elwise(side_inds, make_side_quad);
    });
    return lookup_table[el_side];
}

template < BasisType BT, mesh::ElementType ET, el_o_t EO, el_o_t BO >
const auto& getBasisAtNodes()
    requires(mesh::isGeomType(ET))
{
    static const auto value = std::invoke([] {
        const auto& ref_loc = mesh::getNodeLocations< ET, EO >();
        return tabulateBasis< BT, ET, BO >(std::span{ref_loc});
    });
    return value;
}

template < BasisType BT, mesh::ElementType ET, el_o_t EO, el_o_t BO >
const auto& getBasisAtSideNodes(el_side_t side)
    requires(mesh::isGeomType(ET))
{
    using eltraits                 = mesh::ElementTraits< mesh::Element< ET, EO > >;
    static const auto lookup_table = std::invoke([] {
        return util::elwise(util::makeIotaArray< el_side_t, eltraits::n_sides >(), [](auto i) {
            const auto ref_loc = mesh::getSideNodeLocations< ET, EO >(i);
            return tabulateBasis< BT, ET, BO >(std::span{ref_loc});
        });
    });
    return lookup_table[side];
}

template < size_t num_approx_bases, size_t num_geom_bases, size_t num_points, dim_t dim >
struct TabulatedBasisAtPointsView
{
    const TabulatedBasis< num_approx_bases, num_points, dim >* approx_basis;
    const TabulatedBasis< num_geom_bases, num_points, dim >*   geom_basis;
};

template < size_t num_approx_bases, size_t num_geom_bases, size_t num_points, dim_t dim >
struct TabulatedBasisAtQuadratureView
{
    TabulatedBasisAtPointsView< num_approx_bases, num_geom_bases, num_points, dim > bases;
    std::span< const val_t, num_points >                                            weights;
};

template < BasisType BT, mesh::ElementType ET, el_o_t EO, quad::QuadratureType QT, q_o_t QO >
auto getQuadratureView()
{
    using eltraits_t   = mesh::ElementTraits< mesh::Element< ET, EO > >;
    constexpr auto GT  = eltraits_t::geom_type;
    constexpr auto GO  = eltraits_t::geom_order;
    constexpr auto GBT = BasisType::Lagrange;

    const auto approx_basis_ptr = &getBasisInDomain< BT, GT, EO, QT, QO >();
    const auto geom_basis_ptr   = &getBasisInDomain< GBT, GT, GO, QT, QO >();
    const auto weights          = std::span{quad::getQuadrature< QT, QO, GT >().weights};
    const auto bases            = TabulatedBasisAtPointsView{approx_basis_ptr, geom_basis_ptr};
    return TabulatedBasisAtQuadratureView{.bases = bases, .weights = weights};
}

template < BasisType BT, mesh::ElementType ET, el_o_t EO, quad::QuadratureType QT, q_o_t QO >
auto getSideQuadratureView(el_side_t side)
{
    using eltraits_t  = mesh::ElementTraits< mesh::Element< ET, EO > >;
    constexpr auto GT = eltraits_t::geom_type;
    constexpr auto GO = eltraits_t::geom_order;

    const auto     approx_basis_ptr = &getBasisAtBoundary< BT, GT, EO, QT, QO >(side);
    const auto     geom_basis_ptr   = &getBasisAtBoundary< BT, GT, GO, QT, QO >(side);
    constexpr auto quad_size        = std::decay_t< decltype(*approx_basis_ptr) >::size;
    using span_t                    = std::span< const val_t, quad_size >;
    const auto weights              = std::invoke([] -> span_t {
        using enum mesh::ElementType;
        static constexpr auto point_wgt_array = std::array{val_t{}};
        if constexpr (GT == Line)
            return span_t{point_wgt_array};
        else
        {
            constexpr auto boundary_type = GT == Hex ? Quad : Line;
            return span_t{quad::getQuadrature< QT, QO, boundary_type >().weights};
        }
    });
    const auto bases                = TabulatedBasisAtPointsView{approx_basis_ptr, geom_basis_ptr};
    return TabulatedBasisAtQuadratureView{.bases = bases, .weights = weights};
}

template < BasisType BT, mesh::ElementType ET, el_o_t EO >
auto getNodeBasisView()
{
    using eltraits_t   = mesh::ElementTraits< mesh::Element< ET, EO > >;
    constexpr auto GT  = eltraits_t::geom_type;
    constexpr auto GO  = eltraits_t::geom_order;
    constexpr auto GBT = BasisType::Lagrange;

    const auto approx_basis_ptr = &getBasisAtNodes< BT, GT, EO, EO >();
    const auto geom_basis_ptr   = &getBasisAtNodes< GBT, GT, EO, GO >();
    return TabulatedBasisAtPointsView{approx_basis_ptr, geom_basis_ptr};
}

template < BasisType BT, mesh::ElementType ET, el_o_t EO >
auto getSideNodeBasisView(el_side_t side)
{
    using eltraits_t   = mesh::ElementTraits< mesh::Element< ET, EO > >;
    constexpr auto GT  = eltraits_t::geom_type;
    constexpr auto GO  = eltraits_t::geom_order;
    constexpr auto GBT = BasisType::Lagrange;

    const auto approx_basis_ptr = &getBasisAtSideNodes< BT, GT, EO, EO >(side);
    const auto geom_basis_ptr   = &getBasisAtSideNodes< GBT, GT, EO, GO >(side);
    return TabulatedBasisAtPointsView{approx_basis_ptr, geom_basis_ptr};
}
} // namespace lstr::basis
#endif // L3STER_BASIS_BASISATQUADRATURE_HPP
