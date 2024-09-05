#ifndef L3STER_QUAD_EVALQUADRATURE_HPP
#define L3STER_QUAD_EVALQUADRATURE_HPP

#include "l3ster/quad/Quadrature.hpp"
#include "l3ster/util/Concepts.hpp"

#include <concepts>
#include <functional>
#include <numeric>
#include <tuple>

namespace lstr::quad
{
namespace detail
{
template < typename T >
concept NoexceptSimplyPlusable_c = requires(T a, T b) {
    { a + b } noexcept -> std::convertible_to< T >;
};

template < typename T, typename Q >
concept QuadIntegrable_c = requires(T fun, Q::q_points_t::value_type point, Q::weights_t::value_type weight) {
    { std::apply(fun, point) * weight } -> NoexceptSimplyPlusable_c;
};

template < typename Integrand, typename Q >
    requires QuadIntegrable_c< Integrand, Q >
struct QuadIntTraits
{
    using pt_t = Q::q_points_t::value_type;
    using w_t  = Q::weights_t::value_type;

    using kernel_t = decltype(std::apply(std::declval< Integrand >(), std::declval< pt_t >()) * std::declval< w_t >());
};

template < typename T, typename Q >
concept QuadKernelDefaultInitializable_c = std::default_initializable< typename QuadIntTraits< T, Q >::kernel_t >;
} // namespace detail

// This is a weighted reduction of `fun` over quadrature points
template < typename Integrand, q_l_t quad_length, dim_t quad_dim >
auto evalQuadrature(Integrand&& integrator, const Quadrature< quad_length, quad_dim >& quad)
    requires detail::QuadIntegrable_c< Integrand, Quadrature< quad_length, quad_dim > > and
             detail::QuadKernelDefaultInitializable_c< Integrand, Quadrature< quad_length, quad_dim > >
{
    using quad_t   = Quadrature< quad_length, quad_dim >;
    using point_t  = quad_t::q_points_t::value_type;
    using weight_t = quad_t::weights_t::value_type;

    const auto invoke_and_weigh = [&](const point_t& point, const weight_t& weight) {
        return std::apply(integrator, point) * weight;
    };
    const auto zero_init = typename detail::QuadIntTraits< Integrand, Quadrature< quad_length, quad_dim > >::kernel_t{};
    return std::transform_reduce(
        quad.points.cbegin(), quad.points.cend(), quad.weights.cbegin(), zero_init, std::plus<>{}, invoke_and_weigh);
}

template < typename Integrand, typename Zero, dim_t dim >
concept QuadIntegrable_c = ReturnInvocable_c< Integrand, Zero, ptrdiff_t, const std::array< val_t, dim > > and
                           requires(Zero zero, val_t wgt) { zero += zero * wgt; };

// This is a weighted reduction of `fun` over point-index pairs. This is meant to enable more efficient
// calculation, where certain quantities (e.g. basis derivatives) can be precomputed collectively for all
// quadrature points, and then accessed by index during quadrature evaluation
template < typename Integrand, q_l_t quad_length, dim_t quad_dim, typename Zero >
auto evalQuadrature(Integrand&& integrand, const Quadrature< quad_length, quad_dim >& quadrature, Zero zero)
    requires QuadIntegrable_c< Integrand, Zero, quad_dim >
{
    for (auto&& [i, quad] : std::views::zip(quadrature.points, quadrature.weights) | std::views::enumerate)
    {
        const auto& [point, weight] = quad;
        const auto integrand_value  = integrand(i, point);
        zero += integrand_value * weight;
    }
    return zero;
}
} // namespace lstr::quad
#endif // L3STER_QUAD_EVALQUADRATURE_HPP
