#ifndef L3STER_ASSEMBLY_COMPUTEPHYSBASESATQPOINTS_HPP
#define L3STER_ASSEMBLY_COMPUTEPHYSBASESATQPOINTS_HPP

#include "l3ster/local_assembly/ComputeRefBasesAtQpoints.hpp"
#include "l3ster/mapping/JacobiMat.hpp"

namespace lstr
{
namespace detail
{
template < ElementTypes ET, el_o_t EO >
using jacobian_t = Eigen::Matrix< val_t, Element< ET, EO >::native_dim, Element< ET, EO >::native_dim >;
} // namespace detail

template < BasisTypes BT, ElementTypes ET, el_o_t EO, q_l_t QL, dim_t QD >
auto computePhysicalBasesAtQpoints(
    const std::array< detail::jacobian_t< ET, EO >, QL >& jacobians,
    const Quadrature< QL, QD >& quadrature) requires(ElementTraits< Element< ET, EO > >::native_dim == QD)
{
    const auto& ref_ders = detail::computeRefBasisDersAtQpoints< BT, ET, EO >(quadrature);
    using ret_t          = std::remove_cvref_t< decltype(ref_ders) >;
    ret_t ret_val; // NOLINT initialization at fill-in - note the innermost loop is unrolled by 1

    using jac_t = typename std::remove_cvref_t< decltype(jacobians) >::value_type;

    for (size_t qp_ind : std::views::iota(0u, QL))
    {
        const jac_t jacobian_inv = jacobians[qp_ind].inverse();
        for (ptrdiff_t der_ind = 0; auto& der : ret_val)
        {
            der(qp_ind, Eigen::all) = ref_ders[0](qp_ind, Eigen::all) * jacobian_inv(der_ind, 0);
            for (ptrdiff_t dim_ind : std::views::iota(1u, Element< ET, EO >::native_dim))
                der(qp_ind, Eigen::all) += ref_ders[dim_ind](qp_ind, Eigen::all) * jacobian_inv(der_ind, dim_ind);
            ++der_ind;
        }
    }

    return ret_val;
}

template < ElementTypes ET, el_o_t EO, q_l_t QL, dim_t QD >
auto computeElementJacobiansAtQpoints(const Element< ET, EO >& element, const Quadrature< QL, QD >& quadrature)
{
    const auto jac_gen = getNatJacobiMatGenerator(element);

    using ret_t = std::array< detail::jacobian_t< ET, EO >, QL >;
    ret_t ret_val; // NOLINT initialization follows immediately below
    std::ranges::transform(quadrature.getPoints(), begin(ret_val), [&](const auto& pc) { return jac_gen(Point{pc}); });
    return ret_val;
}

template < BasisTypes BT, ElementTypes ET, el_o_t EO, q_l_t QL, dim_t QD >
auto computePhysicalBasesAtQpoints(const Element< ET, EO >& element, const Quadrature< QL, QD >& quadrature)
{
    const auto jac_array = computeElementJacobiansAtQpoints(element, quadrature);
    return computePhysicalBasesAtQpoints< BT, ET, EO >(jac_array, quadrature);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_COMPUTEPHYSBASESATQPOINTS_HPP
