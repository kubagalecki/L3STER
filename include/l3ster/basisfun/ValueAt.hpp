#ifndef L3STER_BASISFUN_VALUEAT_HPP
#define L3STER_BASISFUN_VALUEAT_HPP

#include "ReferenceBasisFunction.hpp"

namespace lstr
{
template < BasisTypes BT, ElementTypes T, el_o_t O, random_access_typed_range< val_t > R >
val_t valueAt(const Element< T, O >&, R&& node_vals, const Point< ElementTraits< Element< T, O > >::native_dim >& point)
{
    return [&]< el_locind_t... I >(std::integer_sequence< el_locind_t, I... >)
    {
        const auto compute_phi_ind = [&]< el_locind_t Ind >(std::integral_constant< el_locind_t, Ind >) {
            return node_vals[Ind] * ReferenceBasisFunction< T, O, Ind, BT >{}(point);
        };
        return (compute_phi_ind(std::integral_constant< el_locind_t, I >{}) + ...);
    }
    (std::make_integer_sequence< el_locind_t, Element< T, O >::n_nodes >{});
}
} // namespace lstr
#endif // L3STER_BASISFUN_VALUEAT_HPP