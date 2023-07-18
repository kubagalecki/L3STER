#ifndef L3STER_MESH_NODEPHYSICALLOCATION_HPP
#define L3STER_MESH_NODEPHYSICALLOCATION_HPP

#include "l3ster/mapping/MapReferenceToPhysical.hpp"
#include "l3ster/mesh/NodeReferenceLocation.hpp"

namespace lstr::mesh
{
template < ElementType T, el_o_t O >
auto nodePhysicalLocation(const Element< T, O >& element)
{
    const auto& ref_locs = getNodeLocations< T, O >();
    std::array< Point< 3 >, std::tuple_size_v< std::decay_t< decltype(ref_locs) > > > retval{};
    std::ranges::transform(
        ref_locs, begin(retval), [&](const auto& xi) { return map::mapToPhysicalSpace(element, xi); });
    return retval;
}

template < ElementType T, el_o_t O >
Point< 3 > nodePhysicalLocation(const Element< T, O >& element, el_locind_t i)
{
    return map::mapToPhysicalSpace(element, getNodeLocations< T, O >()[i]);
}
} // namespace lstr::mesh
#endif // L3STER_MESH_NODEPHYSICALLOCATION_HPP
