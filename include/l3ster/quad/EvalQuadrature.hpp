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
concept NoexceptSimplyPlusable_c = requires(T a, T b)
{
    {
        a + b
    }
    noexcept->std::convertible_to< T >;
};

template < typename T, typename Q >
concept QuadIntegrable_c = requires(T fun, Q::q_points_t::value_type point, Q::weights_t::value_type weight)
{
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
template < typename Integrator, q_l_t QLENGTH, dim_t QDIM >
auto evalQuadrature(Integrator&& integrator, const Quadrature< QLENGTH, QDIM >& quad) requires
    detail::QuadIntegrable_c< Integrator, Quadrature< QLENGTH, QDIM > > and
    detail::QuadKernelDefaultInitializable_c< Integrator, Quadrature< QLENGTH, QDIM > >
{
    using quad_t   = Quadrature< QLENGTH, QDIM >;
    using point_t  = quad_t::q_points_t::value_type;
    using weight_t = quad_t::weights_t::value_type;

    const auto invoke_and_weigh = [&](const point_t& point, const weight_t& weight) {
        return std::apply(integrator, point) * weight;
    };
    const auto zero_init = typename detail::QuadIntTraits< Integrator, Quadrature< QLENGTH, QDIM > >::kernel_t{};
    return std::transform_reduce(quad.getPoints().cbegin(),
                                 quad.getPoints().cend(),
                                 quad.getWeights().cbegin(),
                                 zero_init,
                                 std::plus<>{},
                                 invoke_and_weigh);
}

// This is a weighted reduction of `fun` over point-index pairs. This is meant to enable more efficient calculation,
// where certain quantities (e.g. basis derivatives) can be precomputed collectively for all quadrature points, and then
// accessed by index during quadrature evaluation
template < typename Integrator, typename ZeroGenerator, q_l_t QLENGTH, dim_t QDIM >
auto evalQuadrature(Integrator&&                       integrator,
                    const Quadrature< QLENGTH, QDIM >& quad,
                    ZeroGenerator                      zero_gen) noexcept requires
    requires(Integrator                                          integr,
             Quadrature< QLENGTH, QDIM >::q_points_t::value_type qp,
             Quadrature< QLENGTH, QDIM >::weights_t::value_type  w,
             ptrdiff_t                                           index,
             ZeroGenerator                                       zg)
{
    {
        zg()
    }
    noexcept;
    {
        integr(index, qp)
    }
    noexcept;
    requires requires(std::remove_cvref_t< decltype(zg()) > el0) { el0 += integr(index, qp) * w; };
}
{
    auto zero = zero_gen();
    for (ptrdiff_t i = 0; const auto& qp : quad.getPoints())
    {
        zero += integrator(i, qp) * quad.getWeights()[i];
        ++i;
    }
    return zero;
}
} // namespace lstr
#endif // L3STER_QUAD_EVALQUADRATURE_HPP
