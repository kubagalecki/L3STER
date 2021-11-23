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

template < QuadratureTypes QT, q_o_t QO, ElementTypes ET >
inline constexpr auto quad_size = std::remove_cvref_t< decltype(getQuadrature< QT, QO, ET >()) >::size;
} // namespace detail

template < QuadratureTypes QT, q_o_t QO, BasisTypes BT, ElementTypes ET, el_o_t EO >
auto computePhysicalBasesAtQpoints(
    const std::array< detail::jacobian_t< ET, EO >, detail::quad_size< QT, QO, ET > >& jacobians)
{
    const auto& ref_ders = getRefBasisDersAtQpoints< QT, QO, ET, EO, BT >();
    using ret_t          = std::remove_cvref_t< decltype(ref_ders) >;
    ret_t ret_val; // NOLINT initialization at fill-in - note the innermost loop is unrolled by 1

    using jac_t = std::remove_cvref_t< decltype(jacobians) >::value_type;

    const auto& quadrature = getQuadrature< QT, QO, ET >();

    for (size_t qp_ind = 0; qp_ind < quadrature.size; ++qp_ind)
    {
        const jac_t jacobian_inv = jacobians[qp_ind].inverse();
        for (ptrdiff_t der_ind = 0; auto& der : ret_val)
        {
            der(qp_ind, Eigen::all) = ref_ders[0](qp_ind, Eigen::all) * jacobian_inv(der_ind, 0);
            for (ptrdiff_t i = 1; i < static_cast< ptrdiff_t >(Element< ET, EO >::native_dim); ++i)
                der(qp_ind, Eigen::all) += ref_ders[i](qp_ind, Eigen::all) * jacobian_inv(der_ind, i);
            ++der_ind;
        }
    }

    return ret_val;
}

template < QuadratureTypes QT, q_o_t QO, ElementTypes ET, el_o_t EO >
auto computeElementJacobiansAtQpoints(const Element< ET, EO >& element)
{
    const auto  jac_gen = getNatJacobiMatGenerator(element);
    const auto& quad    = getQuadrature< QT, QO, ET >();

    using ret_t = std::array< detail::jacobian_t< ET, EO >, quad.size >;
    ret_t ret_val; // NOLINT initialization follows immediately below
    std::ranges::transform(quad.getPoints(), begin(ret_val), [&](const auto& pc) { return jac_gen(Point{pc}); });
    return ret_val;
}

template < QuadratureTypes QT, q_o_t QO, BasisTypes BT, ElementTypes ET, el_o_t EO >
auto computePhysicalBasesAtQpoints(const Element< ET, EO >& element)
{
    const auto jac_array = computeElementJacobiansAtQpoints< QT, QO >(element);
    return computePhysicalBasesAtQpoints< QT, QO, BT, ET, EO >(jac_array);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_COMPUTEPHYSBASESATQPOINTS_HPP
