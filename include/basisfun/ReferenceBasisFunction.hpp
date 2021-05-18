#ifndef L3STER_BASISFUN_REFERENCEBASISFUNCTION_HPP
#define L3STER_BASISFUN_REFERENCEBASISFUNCTION_HPP

#include "math/LagrangeInterpolation.hpp"
#include "math/LobattoRuleAbsc.hpp"
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
    return lagrangeInterp(lobatto_rule_absc< val_t, O + 1 >, vals);
}

template < el_o_t O, el_locind_t I >
inline const auto line_basis_polynomial = computeLineBasisPolynomial< O, I >();

template < el_o_t O, el_locind_t I >
val_t evaluateLineBasisFun(const Point< 1 >& point)
{
    return line_basis_polynomial< O, I >.evaluate(point.x());
}

template < el_o_t O, el_locind_t I >
val_t evaluateQuadBasisFun(const Point< 2 >& point)
{
    constexpr auto nodes_per_edge = O + 1;
    constexpr auto xi_ind         = I % nodes_per_edge;
    constexpr auto eta_ind        = I / nodes_per_edge;

    return evaluateLineBasisFun< O, xi_ind >(Point{point.x()}) * evaluateLineBasisFun< O, eta_ind >(Point{point.y()});
}

template < el_o_t O, el_locind_t I >
val_t evaluateHexBasisFun(const Point< 3 >& point)
{
    constexpr auto nodes_per_edge = O + 1;
    constexpr auto nodes_per_face = nodes_per_edge * nodes_per_edge;
    constexpr auto xi_eta_ind     = I % nodes_per_face;
    constexpr auto zeta_ind       = I / nodes_per_face;

    return evaluateQuadBasisFun< O, xi_eta_ind >(Point{point.x(), point.y()}) *
           evaluateLineBasisFun< O, zeta_ind >(Point{point.z()});
}
} // namespace detail

template < ElementTypes T, el_o_t O, el_locind_t I >
requires(I < ElementTraits< Element< T, O > >::nodes_per_element) struct ReferenceBasisFunction
{
    val_t operator()(const Point< ElementTraits< Element< T, O > >::native_dim >& point)
    {
        if constexpr (T == ElementTypes::Line)
            return detail::evaluateLineBasisFun< O, I >(point);
        else if constexpr (T == ElementTypes::Quad)
            return detail::evaluateQuadBasisFun< O, I >(point);
        else if constexpr (T == ElementTypes::Hex)
            return detail::evaluateHexBasisFun< O, I >(point);
    }
};
} // namespace lstr
#endif // L3STER_BASISFUN_REFERENCEBASISFUNCTION_HPP
