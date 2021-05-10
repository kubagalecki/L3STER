#ifndef L3STER_MESH_MAPREFERENCETOPHYSICAL_HPP
#define L3STER_MESH_MAPREFERENCETOPHYSICAL_HPP

#include "basisfun/ValueAt.hpp"
#include "mesh/Element.hpp"

#include <cmath>

namespace lstr
{
namespace detail
{
template < el_o_t O, random_access_typed_range< Point< 1 > > R >
auto mapLineToPhysicalSpace(const Element< ElementTypes::Line, O >& element, R&& xi)
{
    using vector_t                      = Point< 3 >::vector_t;
    constexpr auto                 n_xi = range_constexpr_size_v< R >;
    std::array< Point< 3 >, n_xi > retval;

    const vector_t v1    = element.getData().vertices[0];
    const vector_t v2    = element.getData().vertices[1];
    const vector_t v_tan = v2 - v1;

    for (size_t i = 0; const auto& point : xi)
        retval[i++] = Point< 3 >{v1 + v_tan * ((point.x() + 1.) / 2.)};

    return retval;
}

template < el_o_t O, random_access_typed_range< Point< 2 > > R >
auto mapQuadToPhysicalSpace(const Element< ElementTypes::Quad, O >& element, R&& xi)
{
    using o1_el_t = Element< ElementTypes ::Quad, 1 >;
    const o1_el_t dummy_el{o1_el_t::node_array_t{}, element.getData(), 0};
    const auto&   vertices = element.getData().vertices;

    const auto x_coord_view = vertices | std::views::transform([](const Point< 3 >& p) { return p.x(); });
    const auto y_coord_view = vertices | std::views::transform([](const Point< 3 >& p) { return p.y(); });
    const auto z_coord_view = vertices | std::views::transform([](const Point< 3 >& p) { return p.z(); });

    constexpr auto                 n_xi = range_constexpr_size_v< R >;
    std::array< Point< 3 >, n_xi > retval;

    const auto x_vals = valueAt(dummy_el, x_coord_view, xi);
    const auto y_vals = valueAt(dummy_el, y_coord_view, xi);
    const auto z_vals = valueAt(dummy_el, z_coord_view, xi);

    for (size_t i = 0; auto& point : retval)
    {
        point = Point< 3 >{{x_vals[i], y_vals[i], z_vals[i]}};
        ++i;
    }

    return retval;
}
} // namespace detail

template < ElementTypes                                                                       T,
           el_o_t                                                                             O,
           random_access_typed_range< Point< ElementTraits< Element< T, O > >::native_dim > > R >
auto mapToPhysicalSpace(const Element< T, O >& element, R&& points)
{
    if constexpr (T == ElementTypes::Line)
        return detail::mapLineToPhysicalSpace(element, points);
    else if constexpr (T == ElementTypes::Quad)
        return detail::mapQuadToPhysicalSpace(element, points);
    else if constexpr (T == ElementTypes::Hex)
    {}
}
} // namespace lstr
#endif // L3STER_MESH_MAPREFERENCETOPHYSICAL_HPP
