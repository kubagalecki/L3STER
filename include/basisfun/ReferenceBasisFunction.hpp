#ifndef L3STER_BASISFUN_REFERENCEBASISFUNCTION_HPP
#define L3STER_BASISFUN_REFERENCEBASISFUNCTION_HPP

#include "math/ComputeLobattoRuleAbsc.hpp"
#include "math/LagrangeInterpolation.hpp"
#include "mesh/ElementTraits.hpp"
#include "mesh/Point.hpp"

namespace lstr
{
namespace detail
{
template < el_o_t O, el_locind_t I >
auto computeLineBasisPolynomial()
{
    constexpr auto vals = [] {
        std::array< val_t, O + 1 > retval{};
        retval[I] = 1.;
        return retval;
    }();
    return lagrangeInterp(computeLobattoRuleAbsc< val_t, O + 1 >(), vals);
}

template < el_o_t O, el_locind_t I >
inline const auto line_basis_polynomial = computeLineBasisPolynomial< O, I >();

template < el_o_t O, el_locind_t I, std::ranges::random_access_range R, typename Out >
void evaluateLineBasisFun(R&& points, Out it)
{
    std::ranges::transform(
        points, it, [&](const Point< 1 >& pt) { return line_basis_polynomial< O, I >.evaluate(pt.x()); });
}

template < el_o_t O, el_locind_t I >
val_t evaluateLineBasisFun(const Point< 1 >& point)
{
    return line_basis_polynomial< O, I >.evaluate(point.x());
}

template < el_o_t O, el_locind_t I, std::ranges::random_access_range R, typename Out >
void evaluateQuadBasisFun(R&& points, Out it)
{
    constexpr auto nodes_per_edge = O + 1;
    constexpr auto xi_ind         = I % nodes_per_edge;
    constexpr auto eta_ind        = I / nodes_per_edge;

    std::ranges::transform(points, it, [&](Point< 2 > p) {
        const auto f_xi_val  = evaluateLineBasisFun< O, xi_ind >(Point< 1 >{{p.x()}});
        const auto f_eta_val = evaluateLineBasisFun< O, eta_ind >(Point< 1 >{{p.y()}});
        return f_xi_val * f_eta_val;
    });
}

template < el_o_t O, el_locind_t I >
val_t evaluateQuadBasisFun(const Point< 2 >& point)
{
    val_t retval;
    evaluateQuadBasisFun< O, I >(std::views::single(point), &retval);
    return retval;
}

template < el_o_t O, el_locind_t I, std::ranges::random_access_range R, typename Out >
void evaluateHexBasisFun(R&& points, Out it)
{
    constexpr auto nodes_per_edge = O + 1;
    constexpr auto nodes_per_face = nodes_per_edge * nodes_per_edge;
    constexpr auto xi_eta_ind     = I % nodes_per_face;
    constexpr auto zeta_ind       = I / nodes_per_face;

    std::ranges::transform(points, it, [&](Point< 3 > p) {
        const auto f_xi_eta_val = evaluateQuadBasisFun< O, xi_eta_ind >(Point< 2 >{p.x(), p.y()});
        const auto f_zeta_val   = evaluateLineBasisFun< O, zeta_ind >({p.z()});
        return f_xi_eta_val * f_zeta_val;
    });
}
} // namespace detail

template < ElementTypes T, el_o_t O, el_locind_t I >
requires(I < ElementTraits< Element< T, O > >::nodes_per_element) struct ReferenceBasisFunction
{
    template < random_access_typed_range< Point< ElementTraits< Element< T, O > >::native_dim > > R,
               std::weakly_incrementable                                                          Out >
    requires std::indirectly_writable< Out, val_t >
    void operator()(R&& points, Out it)
    {
        if constexpr (T == ElementTypes::Line)
            return detail::evaluateLineBasisFun< O, I >(std::forward< R >(points), it);
        else if constexpr (T == ElementTypes::Quad)
            return detail::evaluateQuadBasisFun< O, I >(std::forward< R >(points), it);
        else if constexpr (T == ElementTypes::Hex)
            return detail::evaluateHexBasisFun< O, I >(std::forward< R >(points), it);
    }
};
} // namespace lstr
#endif // L3STER_BASISFUN_REFERENCEBASISFUNCTION_HPP
