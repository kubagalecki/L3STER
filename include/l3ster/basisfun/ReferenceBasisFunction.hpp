#ifndef L3STER_BASISFUN_REFERENCEBASISFUNCTION_HPP
#define L3STER_BASISFUN_REFERENCEBASISFUNCTION_HPP

#include "l3ster/math/LagrangeInterpolation.hpp"
#include "l3ster/math/LobattoRuleAbsc.hpp"
#include "l3ster/mesh/Element.hpp"
#include "l3ster/mesh/ElementTraits.hpp"
#include "l3ster/mesh/Point.hpp"

namespace lstr
{
enum struct BasisTypes
{
    Lagrange
};

enum struct DerDim : dim_t
{
    NoDer = 0,
    DX1   = 1,
    DX2   = 2,
    DX3   = 3
};

namespace detail
{
template < el_o_t O, el_locind_t I >
auto computeLagrangeLineBasisPolynomial()
{
    constexpr auto vals = [] {
        std::array< val_t, O + 1 > retval{};
        retval[I] = 1.;
        return retval;
    }();
    return lagrangeInterp(lobatto_rule_absc< val_t, O + 1 >, vals);
}

template < el_o_t O, el_locind_t I >
inline const auto lagrange_line_basis_polynomial = computeLagrangeLineBasisPolynomial< O, I >();

template < el_o_t O, el_locind_t I >
inline const auto lagrange_line_basis_polynomial_der = lagrange_line_basis_polynomial< O, I >.derivative();

template < el_o_t O, el_locind_t I >
val_t evaluateLagrangeLineBasisFun(const Point< 1 >& point)
{
    return lagrange_line_basis_polynomial< O, I >.evaluate(point.x());
}

template < el_o_t O, el_locind_t I >
val_t evaluateLagrangeLineBasisFunDer(const Point< 1 >& point)
{
    return lagrange_line_basis_polynomial_der< O, I >.evaluate(point.x());
}

template < el_o_t O, el_locind_t I, BasisTypes BT >
val_t evaluateLineBasisFun(const Point< 1 >& point)
{
    if constexpr (BT == BasisTypes::Lagrange)
        return lagrange_line_basis_polynomial< O, I >.evaluate(point.x());
}

template < el_o_t O, el_locind_t I, BasisTypes BT >
val_t evaluateLineBasisFunDer(const Point< 1 >& point)
{
    if constexpr (BT == BasisTypes::Lagrange)
        return lagrange_line_basis_polynomial_der< O, I >.evaluate(point.x());
}

// based on standard tensor product expansion
template < el_o_t O, el_locind_t I, BasisTypes BT >
val_t evaluateQuadBasisFun(const Point< 2 >& point)
{
    constexpr auto nodes_per_edge = O + 1;
    constexpr auto xi_ind         = I % nodes_per_edge;
    constexpr auto eta_ind        = I / nodes_per_edge;

    return evaluateLineBasisFun< O, xi_ind, BT >(Point{point.x()}) *
           evaluateLineBasisFun< O, eta_ind, BT >(Point{point.y()});
}

template < el_o_t O, el_locind_t I, BasisTypes BT >
val_t evaluateQuadBasisFunDerXi(const Point< 2 >& point)
{
    constexpr auto nodes_per_edge = O + 1;
    constexpr auto xi_ind         = I % nodes_per_edge;
    constexpr auto eta_ind        = I / nodes_per_edge;

    return evaluateLineBasisFunDer< O, xi_ind, BT >(Point{point.x()}) *
           evaluateLineBasisFun< O, eta_ind, BT >(Point{point.y()});
}

template < el_o_t O, el_locind_t I, BasisTypes BT >
val_t evaluateQuadBasisFunDerEta(const Point< 2 >& point)
{
    constexpr auto nodes_per_edge = O + 1;
    constexpr auto xi_ind         = I % nodes_per_edge;
    constexpr auto eta_ind        = I / nodes_per_edge;

    return evaluateLineBasisFun< O, xi_ind, BT >(Point{point.x()}) *
           evaluateLineBasisFunDer< O, eta_ind, BT >(Point{point.y()});
}

template < el_o_t O, el_locind_t I, BasisTypes BT >
val_t evaluateHexBasisFun(const Point< 3 >& point)
{
    constexpr auto nodes_per_edge = O + 1;
    constexpr auto nodes_per_face = nodes_per_edge * nodes_per_edge;
    constexpr auto xi_eta_ind     = I % nodes_per_face;
    constexpr auto zeta_ind       = I / nodes_per_face;

    return evaluateQuadBasisFun< O, xi_eta_ind, BT >(Point{point.x(), point.y()}) *
           evaluateLineBasisFun< O, zeta_ind, BT >(Point{point.z()});
}

template < el_o_t O, el_locind_t I, BasisTypes BT >
val_t evaluateHexBasisFunDerXi(const Point< 3 >& point)
{
    constexpr auto nodes_per_edge = O + 1;
    constexpr auto nodes_per_face = nodes_per_edge * nodes_per_edge;
    constexpr auto xi_eta_ind     = I % nodes_per_face;
    constexpr auto zeta_ind       = I / nodes_per_face;

    return evaluateQuadBasisFunDerXi< O, xi_eta_ind, BT >(Point{point.x(), point.y()}) *
           evaluateLineBasisFun< O, zeta_ind, BT >(Point{point.z()});
}

template < el_o_t O, el_locind_t I, BasisTypes BT >
val_t evaluateHexBasisFunDerEta(const Point< 3 >& point)
{
    constexpr auto nodes_per_edge = O + 1;
    constexpr auto nodes_per_face = nodes_per_edge * nodes_per_edge;
    constexpr auto xi_eta_ind     = I % nodes_per_face;
    constexpr auto zeta_ind       = I / nodes_per_face;

    return evaluateQuadBasisFunDerEta< O, xi_eta_ind, BT >(Point{point.x(), point.y()}) *
           evaluateLineBasisFun< O, zeta_ind, BT >(Point{point.z()});
}

template < el_o_t O, el_locind_t I, BasisTypes BT >
val_t evaluateHexBasisFunDerZeta(const Point< 3 >& point)
{
    constexpr auto nodes_per_edge = O + 1;
    constexpr auto nodes_per_face = nodes_per_edge * nodes_per_edge;
    constexpr auto xi_eta_ind     = I % nodes_per_face;
    constexpr auto zeta_ind       = I / nodes_per_face;

    return evaluateQuadBasisFun< O, xi_eta_ind, BT >(Point{point.x(), point.y()}) *
           evaluateLineBasisFunDer< O, zeta_ind, BT >(Point{point.z()});
}
} // namespace detail

template < ElementTypes T, el_o_t O, el_locind_t I, BasisTypes BT, DerDim DER_DIM = DerDim::NoDer >
requires(I < Element< T, O >::n_nodes and
         static_cast< dim_t >(DER_DIM) <= ElementTraits< Element< T, O > >::native_dim) struct ReferenceBasisFunction
{
    val_t operator()(const Point< ElementTraits< Element< T, O > >::native_dim >& point) const
        requires(DER_DIM == DerDim::NoDer)
    {
        if constexpr (T == ElementTypes::Line)
            return detail::evaluateLineBasisFun< O, I, BT >(point);
        else if constexpr (T == ElementTypes::Quad)
            return detail::evaluateQuadBasisFun< O, I, BT >(point);
        else if constexpr (T == ElementTypes::Hex)
            return detail::evaluateHexBasisFun< O, I, BT >(point);
    }

    val_t operator()(const Point< ElementTraits< Element< T, O > >::native_dim >& point) const
        requires(DER_DIM == DerDim::DX1)
    {
        if constexpr (T == ElementTypes::Line)
            return detail::evaluateLineBasisFunDer< O, I, BT >(point);
        else if constexpr (T == ElementTypes::Quad)
            return detail::evaluateQuadBasisFunDerXi< O, I, BT >(point);
        else if constexpr (T == ElementTypes::Hex)
            return detail::evaluateHexBasisFunDerXi< O, I, BT >(point);
    }

    val_t operator()(const Point< ElementTraits< Element< T, O > >::native_dim >& point) const
        requires(DER_DIM == DerDim::DX2)
    {
        if constexpr (T == ElementTypes::Quad)
            return detail::evaluateQuadBasisFunDerEta< O, I, BT >(point);
        else if constexpr (T == ElementTypes::Hex)
            return detail::evaluateHexBasisFunDerEta< O, I, BT >(point);
    }

    val_t operator()(const Point< ElementTraits< Element< T, O > >::native_dim >& point) const
        requires(DER_DIM == DerDim::DX3)
    {
        if constexpr (T == ElementTypes::Hex)
            return detail::evaluateHexBasisFunDerZeta< O, I, BT >(point);
    }
};

namespace detail
{
constexpr DerDim derivativeByIndex(dim_t d)
{
    return static_cast< DerDim >(d + 1);
}
} // namespace detail

template < ElementTypes T, el_o_t O, BasisTypes BT >
auto computeRefBasisDers(const Point< ElementTraits< Element< T, O > >::native_dim >& point)
{
    constexpr dim_t       nat_dim     = ElementTraits< Element< T, O > >::native_dim;
    constexpr el_locind_t n_basis_fun = Element< T, O >::n_nodes;
    using ret_t                       = Eigen::Matrix< val_t, nat_dim, n_basis_fun >;
    ret_t retval; // NOLINT we want raw memory to be written to below
    forConstexpr(
        [&]< el_locind_t I >(std::integral_constant< el_locind_t, I >) {
            forConstexpr(
                [&]< dim_t D >(std::integral_constant< dim_t, D >) {
                    retval(D, I) = ReferenceBasisFunction< T, O, I, BT, detail::derivativeByIndex(D) >{}(point);
                },
                std::make_integer_sequence< dim_t, nat_dim >{});
        },
        std::make_integer_sequence< el_locind_t, n_basis_fun >{});
    return retval;
}
} // namespace lstr
#endif // L3STER_BASISFUN_REFERENCEBASISFUNCTION_HPP