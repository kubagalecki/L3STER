#ifndef L3STER_QUADRATURE_REFERENCEQUADRATURE_HPP
#define L3STER_QUADRATURE_REFERENCEQUADRATURE_HPP

#include "Eigen/Dense"

#include "Quadrature.hpp"
#include "l3ster/defs/Typedefs.h"
#include "l3ster/math/ComputeGaussRule.hpp"
#include "l3ster/quad/QuadratureTypes.h"

#include <algorithm>
#include <cmath>

namespace lstr
{
template < QuadratureTypes QT, q_o_t QO >
constexpr auto getRefQuadSize()
{
    if constexpr (QT == QuadratureTypes::GLeg)
        return QO / 2 + 1;
}

template < QuadratureTypes QT, q_o_t QO >
const auto& getReferenceQuadrature() requires(QT == QuadratureTypes::GLeg)
{
    static const auto value = [] {
        constexpr size_t size = getRefQuadSize< QT, QO >();
        using quadrature_t    = Quadrature< size, 1 >;

        constexpr auto a = [](size_t x) {
            return static_cast< val_t >(2u * x - 1u) / static_cast< val_t >(x);
        };
        constexpr auto b = [](size_t) {
            return 0.;
        };
        constexpr auto c = [](size_t x) {
            return static_cast< val_t >(x - 1u) / static_cast< val_t >(x);
        };
        const auto& [qp, w] = computeGaussRule< size >(a, b, c);

        typename quadrature_t::q_points_t q_points;
        typename quadrature_t::weights_t  weights;
        std::ranges::transform(qp, q_points.begin(), [](val_t v) { return std::array{v}; });
        std::ranges::copy(w, weights.begin());

        return quadrature_t{q_points, weights};
    }();
    return value;
}
} // namespace lstr
#endif // L3STER_QUADRATURE_REFERENCEQUADRATURE_HPP
