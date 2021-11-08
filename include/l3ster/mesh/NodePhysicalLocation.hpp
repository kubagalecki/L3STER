#ifndef L3STER_MESH_NODEPHYSICALLOCATION_HPP
#define L3STER_MESH_NODEPHYSICALLOCATION_HPP

#include "NodeReferenceLocation.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"

namespace lstr
{
template < ElementTypes T, el_o_t O >
auto nodePhysicalLocation(const Element< T, O >& element)
{
    const auto& ref_locs = getNodeLocations< T, O >();
    std::array< Point< 3 >, std::tuple_size_v< std::decay_t< decltype(ref_locs) > > > retval; // NOLINT
    std::ranges::transform(ref_locs, begin(retval), [&](const auto& xi) {
        return mapToPhysicalSpace< BasisTypes::Lagrange >(element, xi);
    });
    return retval;
}

template < ElementTypes T, el_o_t O >
Point< 3 > nodePhysicalLocation(const Element< T, O >& element, el_locind_t i)
{
    return mapToPhysicalSpace< BasisTypes::Lagrange >(element, getNodeLocations< T, O >()[i]);
}
} // namespace lstr
#endif // L3STER_MESH_NODEPHYSICALLOCATION_HPP
