#ifndef L3STER_QUAD_EVALQUADRATURE_HPP
#define L3STER_QUAD_EVALQUADRATURE_HPP

#include "Quadrature.hpp"

#include <concepts>
#include <numeric>
#include <tuple>

namespace lstr
{
namespace detail
{
template < typename T >
concept NoexceptSimplyPlusable_c = requires(T a, T b) {
    {
        a + b
    } noexcept -> std::convertible_to< T >;
};

template < typename T, typename Q >
concept QuadIntegrable_c = requires(T fun, Q::q_points_t::value_type point, Q::weights_t::value_type weight) {
    {
        std::apply(fun, point) * weight
    } -> NoexceptSimplyPlusable_c;
};

template < typename Integrator, typename Q >
    requires QuadIntegrable_c< Integrator, Q >
struct QuadIntTraits
{
    using pt_t = Q::q_points_t::value_type;
    using w_t  = Q::weights_t::value_type;

    using kernel_t = decltype(std::apply(std::declval< Integrator >(), std::declval< pt_t >()) * std::declval< w_t >());
};

template < typename T, typename Q >
concept QuadKernelDefaultInitializable_c = std::default_initializable< typename QuadIntTraits< T, Q >::kernel_t >;
} // namespace detail

// This is a weighted reduction of `fun` over quadrature points
template < typename Integrator, q_l_t quad_length, dim_t quad_dim >
auto evalQuadrature(Integrator&& integrator, const Quadrature< quad_length, quad_dim >& quad)
    requires detail::QuadIntegrable_c< Integrator, Quadrature< quad_length, quad_dim > > and
             detail::QuadKernelDefaultInitializable_c< Integrator, Quadrature< quad_length, quad_dim > >
{
    using quad_t   = Quadrature< quad_length, quad_dim >;
    using point_t  = quad_t::q_points_t::value_type;
    using weight_t = quad_t::weights_t::value_type;

    const auto invoke_and_weigh = [&](const point_t& point, const weight_t& weight) {
        return std::apply(integrator, point) * weight;
    };
    const auto zero_init =
        typename detail::QuadIntTraits< Integrator, Quadrature< quad_length, quad_dim > >::kernel_t{};
    return std::transform_reduce(
        quad.points.cbegin(), quad.points.cend(), quad.weights.cbegin(), zero_init, std::plus<>{}, invoke_and_weigh);
}

// This is a weighted reduction of `fun` over point-index pairs. This is meant to enable more efficient calculation,
// where certain quantities (e.g. basis derivatives) can be precomputed collectively for all quadrature points, and then
// accessed by index during quadrature evaluation
template < q_l_t quad_length, dim_t quad_dim >
auto evalQuadrature(auto&& integrator, const Quadrature< quad_length, quad_dim >& quad, auto zero) noexcept
    requires requires(ptrdiff_t index) {
        {
            integrator(index, quad.points[index])
        } noexcept;
        requires requires { zero += integrator(index, quad.points[index]) * quad.weights[index]; };
    }
{
    for (ptrdiff_t i = 0; const auto& qp : quad.points)
    {
        zero += integrator(i, qp) * quad.weights[i];
        ++i;
    }
    return zero;
}
} // namespace lstr
#endif // L3STER_QUAD_EVALQUADRATURE_HPP
