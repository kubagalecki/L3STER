#ifndef L3STER_BASISFUN_VALUEAT_HPP
#define L3STER_BASISFUN_VALUEAT_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"

namespace lstr
{
template < basis::BasisType BT, mesh::ElementType T, el_o_t O, RandomAccessRangeOf< val_t > R >
val_t valueAt(R&& node_vals, const Point< mesh::Element< T, O >::native_dim >& point)
{
    const auto do_index = [&]< el_locind_t Ind >(std::integral_constant< el_locind_t, Ind >) {
        return node_vals[Ind] * basis::ReferenceBasisFunction< T, O, Ind, BT >{}(point);
    };
    const auto do_all_inds = [&]< el_locind_t... I >(std::integer_sequence< el_locind_t, I... >) {
        return (std::invoke(do_index, std::integral_constant< el_locind_t, I >{}) + ...);
    };
    return std::invoke(do_all_inds, std::make_integer_sequence< el_locind_t, mesh::Element< T, O >::n_nodes >{});
}
} // namespace lstr
#endif // L3STER_BASISFUN_VALUEAT_HPP
