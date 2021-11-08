#ifndef L3STER_ASSEMBLY_COMPUTEREFBASESATQPOINTS_HPP
#define L3STER_ASSEMBLY_COMPUTEREFBASESATQPOINTS_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"
#include "l3ster/quad/GenerateQuadrature.hpp"

namespace lstr
{
namespace detail
{
template < QuadratureTypes QT, q_o_t QO, ElementTypes ET, el_o_t EO, BasisTypes BT >
auto computeRefBasesAtQpoints()
{
    const auto     quad    = generateQuadrature< QT, QO, ET >();
    constexpr auto n_qp    = quad.size;
    constexpr auto n_bases = Element< ET, EO >::n_nodes;
    using ret_t            = Eigen::Matrix< val_t, n_qp, n_bases >;

    ret_t ret_val;
    for (ptrdiff_t index = 0; const auto& qp : quad.getPoints())
    {
        const auto val               = computeRefBasis< ET, EO, BT >(Point{qp});
        ret_val(index++, Eigen::all) = val;
    }
    return ret_val;
}

template < QuadratureTypes QT, q_o_t QO, ElementTypes ET, el_o_t EO, BasisTypes BT >
auto computeRefBasisDersAtQpoints()
{
    const auto     quad    = generateQuadrature< QT, QO, ET >();
    constexpr auto n_qp    = quad.size;
    constexpr auto n_bases = Element< ET, EO >::n_nodes;
    constexpr auto nat_dim = ElementTraits< Element< ET, EO > >::native_dim;
    using values_at_qp_t   = Eigen::Matrix< val_t, n_qp, n_bases >;
    using ret_t            = std::array< values_at_qp_t, nat_dim >;

    ret_t ret_val;
    for (ptrdiff_t index = 0; const auto& qp : quad.getPoints())
    {
        const auto ders = computeRefBasisDers< ET, EO, BT >(Point{qp});
        for (ptrdiff_t d = 0; d < nat_dim; ++d)
            ret_val[d](index, Eigen::all) = ders(d, Eigen::all);
        ++index;
    }
    return ret_val;
}
} // namespace detail

template < QuadratureTypes QT, q_o_t QO, ElementTypes ET, el_o_t EO, BasisTypes BT >
const auto& getRefBasesAtQpoints()
{
    static const auto value = detail::computeRefBasesAtQpoints< QT, QO, ET, EO, BT >();
    return value;
}

template < QuadratureTypes QT, q_o_t QO, ElementTypes ET, el_o_t EO, BasisTypes BT >
const auto& getRefBasisDersAtQpoints()
{
    static const auto value = detail::computeRefBasisDersAtQpoints< QT, QO, ET, EO, BT >();
    return value;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_COMPUTEREFBASESATQPOINTS_HPP
