#ifndef L3STER_MESH_MAPREFERENCETOPHYSICAL_HPP
#define L3STER_MESH_MAPREFERENCETOPHYSICAL_HPP

#include "l3ster/basisfun/ValueAt.hpp"
#include "l3ster/mesh/Element.hpp"

#include <cmath>

namespace lstr::map
{
template < mesh::ElementType T, el_o_t O >
auto mapToPhysicalSpace(const mesh::Element< T, O >&                            element,
                        const mesh::Point< mesh::Element< T, O >::native_dim >& point) -> mesh::Point< 3 >
    requires(util::contains({mesh::ElementType::Line, mesh::ElementType::Quad, mesh::ElementType::Hex}, T))
{
    const mesh::Element< T, 1 > o1_el{{}, element.getData(), {}};
    const auto&                 vertices = element.getData().vertices;

    const auto compute_dim = [&](ptrdiff_t dim) {
        return valueAt< basis::BasisType::Lagrange >(
            o1_el, vertices | std::views::transform([&](const mesh::Point< 3 >& p) { return p[dim]; }), point);
    };

    return mesh::Point{compute_dim(0), compute_dim(1), compute_dim(2)};
}
} // namespace lstr::map
#endif // L3STER_MESH_MAPREFERENCETOPHYSICAL_HPP
