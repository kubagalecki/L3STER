#ifndef L3STER_BASISFUN_REFERENCEBASISATNODES_HPP
#define L3STER_BASISFUN_REFERENCEBASISATNODES_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"
#include "l3ster/mesh/NodeReferenceLocation.hpp"

namespace lstr::basis
{
template < mesh::ElementType ET, el_o_t EO, BasisType BT = BasisType::Lagrange >
const auto& getBasisAtNodes()
{
    static const ReferenceBasisAtPoints< ET, EO, mesh::Element< ET, EO >::n_nodes > retval = std::invoke([] {
        const auto& reference_locations = mesh::getNodeLocations< ET, EO, BT >();
        return detail::evalRefBasisAtPoints< BT, ET, EO >(reference_locations);
    });
    return retval;
}
} // namespace lstr::basis
#endif // L3STER_BASISFUN_REFERENCEBASISATNODES_HPP
