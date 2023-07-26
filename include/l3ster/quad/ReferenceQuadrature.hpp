#ifndef L3STER_QUADRATURE_REFERENCEQUADRATURE_HPP
#define L3STER_QUADRATURE_REFERENCEQUADRATURE_HPP

#include "l3ster/common/Typedefs.h"
#include "l3ster/math/ComputeGaussRule.hpp"
#include "l3ster/quad/Quadrature.hpp"

#include <algorithm>
#include <cmath>

namespace lstr::quad
{
consteval size_t getRefQuadSize(QuadratureType QT, q_o_t QO)
{
    switch (QT)
    {
    case QuadratureType::GaussLegendre:
        return QO / 2 + 1;
        break;
    default:
        throw "Reference quadrature size is unknown for this quadrature type";
    }
}

template < QuadratureType QT, q_o_t QO >
const auto& getReferenceQuadrature()
    requires(QT == QuadratureType::GaussLegendre)
{
    static const auto value = std::invoke([] {
        constexpr auto size = getRefQuadSize(QT, QO);
        using quadrature_t  = Quadrature< size, 1 >;

        constexpr auto a = [](size_t x) {
            return static_cast< val_t >(2u * x - 1u) / static_cast< val_t >(x);
        };
        constexpr auto b = [](size_t) {
            return 0.;
        };
        constexpr auto c = [](size_t x) {
            return static_cast< val_t >(x - 1u) / static_cast< val_t >(x);
        };
        const auto [qp, w] = math::computeGaussRule(a, b, c, std::integral_constant< size_t, size >{});

        typename quadrature_t::q_points_t q_points;
        typename quadrature_t::weights_t  weights;
        std::ranges::transform(qp, q_points.begin(), [](val_t v) { return std::array{v}; });
        std::ranges::copy(w, weights.begin());

        return quadrature_t{q_points, weights};
    });
    return value;
}
} // namespace lstr::quad
#endif // L3STER_QUADRATURE_REFERENCEQUADRATURE_HPP
