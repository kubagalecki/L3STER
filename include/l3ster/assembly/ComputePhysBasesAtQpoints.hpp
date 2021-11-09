#ifndef L3STER_ASSEMBLY_COMPUTEPHYSBASESATQPOINTS_HPP
#define L3STER_ASSEMBLY_COMPUTEPHYSBASESATQPOINTS_HPP

#include "l3ster/assembly/ComputeRefBasesAtQpoints.hpp"
#include "l3ster/mapping/JacobiMat.hpp"

namespace lstr
{
template < QuadratureTypes QT, q_o_t QO, BasisTypes BT, ElementTypes ET, el_o_t EO >
auto computePhysicalBasesAtQpoints(const Element< ET, EO >& element)
{
    const auto& ref_ders = getRefBasisDersAtQpoints< QT, QO, ET, EO, BT >();
    using ret_t          = std::remove_cvref_t< decltype(ref_ders) >;
    ret_t ret_val; // NOLINT initialization at fill-in - note the innermost loop is unrolled by 1

    const auto     jacobian_generator = getNatJacobiMatGenerator(element);
    constexpr auto el_dim             = ElementTraits< Element< ET, EO > >::native_dim;
    using jac_t                       = std::invoke_result_t< decltype(jacobian_generator), Point< el_dim > >;

    const auto& quadrature = generateQuadrature< QT, QO, ET >();

    for (ptrdiff_t qp_ind = 0; const auto& qp : quadrature.getPoints())
    {
        const jac_t jacobian_inv = jacobian_generator(Point{qp}).inverse();
        for (ptrdiff_t der_ind = 0; auto& der : ret_val)
        {
            der(qp_ind, Eigen::all) = ref_ders[0](qp_ind, Eigen::all) * jacobian_inv(der_ind, 0);
            for (ptrdiff_t i = 1; i < el_dim; ++i)
                der(qp_ind, Eigen::all) += ref_ders[i](qp_ind, Eigen::all) * jacobian_inv(der_ind, i);
            ++der_ind;
        }
        ++qp_ind;
    }

    return ret_val;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_COMPUTEPHYSBASESATQPOINTS_HPP
