#ifndef L3STER_QUAD_INVOKEQUADRATURE_HPP
#define L3STER_QUAD_INVOKEQUADRATURE_HPP

#include "Quadrature.hpp"

#include <concepts>
#include <numeric>
#include <tuple>

namespace lstr
{
// TODO: write this concept
template < typename T, typename Q >
concept quadrature_integrable = true;

template < typename F, q_l_t QLENGTH, dim_t QDIM >
requires quadrature_integrable< F, Quadrature< QLENGTH, QDIM > >
auto invokeQuadrature(F&& fun, const Quadrature< QLENGTH, QDIM >& quad)
{
    using quad_t   = Quadrature< QLENGTH, QDIM >;
    using point_t  = quad_t::q_points_t::value_type;
    using weight_t = quad_t::weights_t::value_type;

    const auto invoke_and_weigh = [&](const point_t& point, const weight_t& weight) {
        return std::apply(fun, point) * weight;
    };
    const auto zero_init = decltype(invoke_and_weigh(std::declval< point_t >(), std::declval< weight_t >())){};
    return std::transform_reduce(quad.getQPoints().cbegin(),
                                 quad.getQPoints().cend(),
                                 quad.getWeights().cbegin(),
                                 zero_init,
                                 std::plus<>{},
                                 invoke_and_weigh);
}
} // namespace lstr
#endif // L3STER_QUAD_INVOKEQUADRATURE_HPP
