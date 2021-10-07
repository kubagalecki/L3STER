#ifndef L3STER_MESH_MAPREFERENCETOPHYSICAL_HPP
#define L3STER_MESH_MAPREFERENCETOPHYSICAL_HPP

#include "l3ster/basisfun/ValueAt.hpp"
#include "l3ster/mesh/Element.hpp"

#include <cmath>

namespace lstr
{
template < BasisTypes BT, ElementTypes T, el_o_t O >
requires(T == ElementTypes::Line or T == ElementTypes::Quad or T == ElementTypes::Hex) Point< 3 > mapToPhysicalSpace(
    const Element< T, O >& element, const Point< ElementTraits< Element< T, O > >::native_dim >& point)
{
    const Element< T, 1 > o1_el{{}, element.getData(), {}};
    const auto&           vertices = element.getData().vertices;

    const auto compute_dim = [&](ptrdiff_t dim) {
        return valueAt< BT >(
            o1_el, vertices | std::views::transform([&](const Point< 3 >& p) { return p[dim]; }), point);
    };

    return Point{compute_dim(0), compute_dim(1), compute_dim(2)};
}
} // namespace lstr
#endif // L3STER_MESH_MAPREFERENCETOPHYSICAL_HPP
