#ifndef L3STER_BASISFUN_REFERENCEBASISATNODES_HPP
#define L3STER_BASISFUN_REFERENCEBASISATNODES_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"

namespace lstr
{
template < ElementTypes T, el_o_t O >
const auto& getBasisAtNodes()
{
    static const ReferenceBasisAtPoints< T, O, Element< T, O >::n_nodes > retval = std::invoke([] {

    });
    return retval;
}
} // namespace lstr
#endif // L3STER_BASISFUN_REFERENCEBASISATNODES_HPP
