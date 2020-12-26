#ifndef L3STER_QUADRATURE_REFERENCEQUADRATURE_HPP
#define L3STER_QUADRATURE_REFERENCEQUADRATURE_HPP

#include "Eigen/Dense"

#include "math/ComputeGaussRule.hpp"
#include "quad/Quadrature.hpp"
#include "quad/ReferenceQuadratureTraits.hpp"

#include <algorithm>
#include <cmath>

namespace lstr::quad
{
template < QuadratureTypes QTYPE, types::q_o_t QORDER >
struct ReferenceQuadrature;

template < types::q_o_t QORDER >
struct ReferenceQuadrature< QuadratureTypes::GLeg, QORDER >
{
private:
    static auto compute();

public:
    using this_t       = ReferenceQuadrature< QuadratureTypes::GLeg, QORDER >;
    using quadrature_t = Quadrature< ReferenceQuadratureTraits< this_t >::size, 1 >;

    static inline const quadrature_t value = compute();
};

template < types::q_o_t QORDER >
auto ReferenceQuadrature< QuadratureTypes::GLeg, QORDER >::compute()
{
    constexpr auto a = [](size_t x) {
        return static_cast< types::val_t >(2u * x - 1u) / static_cast< types::val_t >(x);
    };
    constexpr auto b = [](size_t) {
        return 0.;
    };
    constexpr auto c = [](size_t x) {
        return static_cast< types::val_t >(x - 1u) / static_cast< types::val_t >(x);
    };
    const auto& [qp, w] =
        math::computeGaussRule< ReferenceQuadratureTraits< this_t >::size >(a, b, c);

    typename quadrature_t::q_points_t q_points;
    typename quadrature_t::weights_t  weights;

    std::transform(
        qp.cbegin(), qp.cend(), q_points.begin(), [](types::val_t v) { return std::array{v}; });
    std::copy(w.cbegin(), w.cend(), weights.begin());

    return quadrature_t{q_points, weights};
}
} // namespace lstr::quad

#endif // L3STER_QUADRATURE_REFERENCEQUADRATURE_HPP
