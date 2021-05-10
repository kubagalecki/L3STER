#ifndef L3STER_BASISFUN_VALUEAT_HPP
#define L3STER_BASISFUN_VALUEAT_HPP

#include "basisfun/ReferenceBasisFunction.hpp"

namespace lstr
{
template < ElementTypes                     T,
           el_o_t                           O,
           std::ranges::random_access_range R1,
           std::ranges::random_access_range R2,
           std::random_access_iterator      Out >
requires std::same_as< std::ranges::range_value_t< R1 >, val_t > &&
    std::same_as< std::ranges::range_value_t< R2 >, Point< ElementTraits< Element< T, O > >::native_dim > >
void valueAt(const Element< T, O >& element, R1&& node_vals, R2&& eval_points, Out out_it)
{
    [&]< el_locind_t... I >(std::integer_sequence< el_locind_t, I... >)
    {
        const auto computeBasisVals = [&]< el_locind_t Ind >(std::integral_constant< el_locind_t, Ind >) {
            const auto ind_vals = ReferenceBasisFunction< T, O, Ind >{}(eval_points);
            for (size_t index = 0; const auto& iv : ind_vals)
                out_it[index++] += iv * node_vals[Ind];
        };
        (computeBasisVals(std::integral_constant< el_locind_t, I >{}), ...);
    }
    (std::make_integer_sequence< el_locind_t, Element< T, O >::n_nodes >{});
}
} // namespace lstr
#endif // L3STER_BASISFUN_VALUEAT_HPP
