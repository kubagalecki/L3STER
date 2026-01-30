#ifndef L3STER_BASISFUN_REFERENCEBASISFUNCTION_HPP
#define L3STER_BASISFUN_REFERENCEBASISFUNCTION_HPP

#include "l3ster/common/Structs.hpp"
#include "l3ster/math/LagrangeInterpolation.hpp"
#include "l3ster/math/LobattoRuleAbsc.hpp"
#include "l3ster/mesh/Element.hpp"
#include "l3ster/mesh/ElementTraits.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Functional.hpp"

namespace lstr::basis
{
enum struct BasisType
{
    Lagrange
};

namespace detail
{
template < std::floating_point T, el_o_t O >
auto computeLagrangeLineBasisPolynomial(size_t i_basis) -> math::Polynomial< T, O >
{
    auto vals        = std::array< T, O + 1 >{};
    vals.at(i_basis) = T{1.};
    auto retval      = math::lagrangeInterp(math::getLobattoRuleAbsc< T, O + 1 >(), vals);
    return retval;
}

template < std::floating_point T, el_o_t O >
auto makeLagrangeBasisTable()
{
    using std::ranges::generate;
    auto retval = std::array< math::Polynomial< T, O >, O + 1 >{};
    generate(retval, [i = 0uz] mutable { return detail::computeLagrangeLineBasisPolynomial< T, O >(i++); });
    return retval;
}

template < std::floating_point T, el_o_t O >
const auto& getLagrangeBasisTable()
{
    // Variable templates initialized in incorrect order for some reason
    static const auto retval = makeLagrangeBasisTable< T, O >();
    return retval;
}

template < std::floating_point T, el_o_t O >
auto computeLagrangeLineBasisPolynomialDerivative(size_t i_basis) -> math::Polynomial< T, O - 1 >
{
    return getLagrangeBasisTable< T, O >().at(i_basis).derivative();
}

template < std::floating_point T, el_o_t O >
auto makeLagrangeBasisDerivativeTable()
{
    using std::ranges::generate;
    auto retval = std::array< math::Polynomial< T, O - 1 >, O + 1 >{};
    generate(retval, [i = 0uz] mutable { return detail::computeLagrangeLineBasisPolynomialDerivative< T, O >(i++); });
    return retval;
}

template < std::floating_point T, el_o_t O >
const auto& getLagrangeBasisDerivativeTable()
{
    // Variable templates initialized in incorrect order for some reason
    static const auto retval = makeLagrangeBasisDerivativeTable< T, O >();
    return retval;
}
} // namespace detail

template < mesh::ElementType T, el_o_t O, BasisType BT >
auto computeReferenceBases(const Point< mesh::Element< T, O >::native_dim >& point)
    requires(mesh::isGeomType(T))
{
    // Only basis currently supported
    static_assert(BT == BasisType::Lagrange);

    constexpr auto n_basis_fun  = static_cast< size_t >(mesh::Element< T, O >::n_nodes);
    constexpr auto n_basis_fun1 = static_cast< size_t >(O + 1);
    constexpr auto basis_inds1  = std::views::iota(0uz, n_basis_fun1);

    // Evaluate in extended precision, cast to val_t after computation
    const auto& basis_table    = detail::getLagrangeBasisTable< long double, O >();
    const auto  tabulate_basis = [&](val_t x) {
        return util::elwise(basis_table, [&](const auto& poly) { return poly.evaluate(x); });
    };

    auto retval = Eigen::Vector< val_t, n_basis_fun >{};
    if constexpr (T == mesh::ElementType::Line)
    {
        for (auto xi : basis_inds1)
            retval[xi] = static_cast< val_t >(basis_table.at(xi).evaluate(point.x()));
    }
    else if constexpr (T == mesh::ElementType::Quad)
    {
        const auto x_vals = tabulate_basis(point.x());
        const auto y_vals = tabulate_basis(point.y());
        for (auto&& [yi, xi] : std::views::cartesian_product(basis_inds1, basis_inds1))
        {
            const auto out_ind = yi * n_basis_fun1 + xi;
            const auto val     = static_cast< val_t >(x_vals.at(xi) * y_vals.at(yi));
            retval[out_ind]    = val;
        }
    }
    else if constexpr (T == mesh::ElementType::Hex)
    {
        const auto x_vals = tabulate_basis(point.x());
        const auto y_vals = tabulate_basis(point.y());
        const auto z_vals = tabulate_basis(point.z());
        for (auto&& [zi, yi, xi] : std::views::cartesian_product(basis_inds1, basis_inds1, basis_inds1))
        {
            const auto out_ind = (zi * n_basis_fun1 + yi) * n_basis_fun1 + xi;
            const auto val     = static_cast< val_t >(x_vals[xi] * y_vals[yi] * z_vals[zi]);
            retval[out_ind]    = val;
        }
    }
    return retval;
}

template < mesh::ElementType T, el_o_t O, BasisType BT >
auto computeReferenceBasisDerivatives(const Point< mesh::Element< T, O >::native_dim >& point)
    requires(mesh::isGeomType(T))
{
    // Only basis currently supported
    static_assert(BT == BasisType::Lagrange);

    constexpr dim_t       nat_dim      = mesh::Element< T, O >::native_dim;
    constexpr el_locind_t n_basis_fun  = mesh::Element< T, O >::n_nodes;
    constexpr el_locind_t n_basis_fun1 = mesh::Element< mesh::ElementType::Line, O >::n_nodes;
    constexpr auto        basis_inds1  = std::views::iota(0uz, n_basis_fun1);

    // Evaluate in extended precision, cast to val_t after computation
    const auto& basis_table     = detail::getLagrangeBasisTable< long double, O >();
    const auto& basis_der_table = detail::getLagrangeBasisDerivativeTable< long double, O >();
    const auto  tabulate_basis  = [&](val_t x) {
        return util::elwise(basis_table, [&](const auto& poly) { return poly.evaluate(x); });
    };
    const auto tabulate_basis_ders = [&](val_t x) {
        return util::elwise(basis_der_table, [&](const auto& poly) { return poly.evaluate(x); });
    };

    auto retval = util::eigen::RowMajorMatrix< val_t, nat_dim, n_basis_fun >{};
    if constexpr (T == mesh::ElementType::Line)
    {
        for (auto xi : basis_inds1)
            retval(0, xi) = static_cast< val_t >(basis_der_table.at(xi).evaluate(point.x()));
    }
    else if constexpr (T == mesh::ElementType::Quad)
    {
        const auto x_vals = tabulate_basis(point.x());
        const auto y_vals = tabulate_basis(point.y());
        const auto x_ders = tabulate_basis_ders(point.x());
        const auto y_ders = tabulate_basis_ders(point.y());
        for (auto&& [yi, xi] : std::views::cartesian_product(basis_inds1, basis_inds1))
        {
            const auto out_ind = yi * n_basis_fun1 + xi;
            const auto dx      = static_cast< val_t >(x_ders[xi] * y_vals[yi]);
            const auto dy      = static_cast< val_t >(x_vals[xi] * y_ders[yi]);
            retval(0, out_ind) = dx;
            retval(1, out_ind) = dy;
        }
    }
    else if constexpr (T == mesh::ElementType::Hex)
    {
        const auto x_vals = tabulate_basis(point.x());
        const auto y_vals = tabulate_basis(point.y());
        const auto z_vals = tabulate_basis(point.z());
        const auto x_ders = tabulate_basis_ders(point.x());
        const auto y_ders = tabulate_basis_ders(point.y());
        const auto z_ders = tabulate_basis_ders(point.z());
        for (auto&& [zi, yi, xi] : std::views::cartesian_product(basis_inds1, basis_inds1, basis_inds1))
        {
            const auto out_ind = (zi * n_basis_fun1 + yi) * n_basis_fun1 + xi;
            const auto dx      = static_cast< val_t >(x_ders[xi] * y_vals[yi] * z_vals[zi]);
            const auto dy      = static_cast< val_t >(x_vals[xi] * y_ders[yi] * z_vals[zi]);
            const auto dz      = static_cast< val_t >(x_vals[xi] * y_vals[yi] * z_ders[zi]);
            retval(0, out_ind) = dx;
            retval(1, out_ind) = dy;
            retval(2, out_ind) = dz;
        }
    }
    return retval;
}
} // namespace lstr::basis
#endif // L3STER_BASISFUN_REFERENCEBASISFUNCTION_HPP
