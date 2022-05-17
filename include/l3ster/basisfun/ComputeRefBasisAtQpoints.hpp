#ifndef L3STER_BASISFUN_COMPUTEREFBASISATQPOINTS_HPP
#define L3STER_BASISFUN_COMPUTEREFBASISATQPOINTS_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"
#include "l3ster/quad/Quadrature.hpp"

namespace lstr
{
template < BasisTypes BT, ElementTypes ET, el_o_t EO, q_l_t QL, dim_t QD >
auto computeRefBasisAtQpoints(const Quadrature< QL, QD >& quad)
    requires(ElementTraits< Element< ET, EO > >::native_dim == QD)
{
    constexpr auto n_bases = Element< ET, EO >::n_nodes;
    using ret_t            = Eigen::Matrix< val_t, QL, n_bases, Eigen::RowMajor >;

    ret_t ret_val;
    for (ptrdiff_t index = 0; const auto& qp : quad.getPoints())
    {
        const auto val               = computeRefBasis< ET, EO, BT >(Point{qp});
        ret_val(index++, Eigen::all) = val;
    }
    return ret_val;
}

template < BasisTypes BT, ElementTypes ET, el_o_t EO, q_l_t QL, dim_t QD >
auto computeRefBasisDersAtQpoints(const Quadrature< QL, QD >& quad)
    requires(ElementTraits< Element< ET, EO > >::native_dim == QD)
{
    constexpr auto n_bases = Element< ET, EO >::n_nodes;
    using values_at_qp_t   = Eigen::Matrix< val_t, QL, n_bases, Eigen::RowMajor >;
    using ret_t            = std::array< values_at_qp_t, Element< ET, EO >::native_dim >;

    ret_t ret_val;
    for (ptrdiff_t index = 0; const auto& qp : quad.getPoints())
    {
        const auto ders = computeRefBasisDers< ET, EO, BT >(Point{qp});
        for (ptrdiff_t d = 0; d < static_cast< ptrdiff_t >(Element< ET, EO >::native_dim); ++d)
            ret_val[d](index, Eigen::all) = ders(d, Eigen::all);
        ++index;
    }
    return ret_val;
}
} // namespace lstr
#endif // L3STER_BASISFUN_COMPUTEREFBASISATQPOINTS_HPP