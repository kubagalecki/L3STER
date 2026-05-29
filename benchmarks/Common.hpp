#ifndef L3STER_BENCHMARKS_COMMON_HPP
#define L3STER_BENCHMARKS_COMMON_HPP

#include "benchmark/benchmark.h"

#include "l3ster/l3ster.hpp"

using namespace lstr;

template < el_o_t O >
constexpr auto getExampleHexElement()
{
    using el_t = mesh::Element< mesh::ElementType::Hex, O >;
    std::array< n_id_t, el_t::n_nodes > nodes;
    n_id_t                              node = 0;
    for (auto i : mesh::ElementTraits< mesh::Element< mesh::ElementType::Hex, O > >::boundary_node_inds)
        nodes.at(i) = node++;
    for (auto i : mesh::ElementTraits< mesh::Element< mesh::ElementType::Hex, O > >::internal_node_inds)
        nodes.at(i) = node++;
    el_t element{nodes,
                 mesh::ElementData< mesh::ElementType::Hex, O >{{Point{0., 0., 0.},
                                                                 Point{1., 0., 0.},
                                                                 Point{0., 1., 0.},
                                                                 Point{1., 1., 0.},
                                                                 Point{0., 0., 1.},
                                                                 Point{1., 0., 1.},
                                                                 Point{0., 1., 1.},
                                                                 Point{2., 2., 2.}}},
                 0};
    return element;
}

template < el_o_t EO >
constexpr auto getExampleHex2Element()
{
    using namespace lstr::mesh;
    constexpr auto ET = ElementType::Hex2;

    auto retval             = Element< ET, EO >{};
    auto& [nodes, data, id] = retval;
    n_id_t n                = 0;
    for (const auto& bnd_node_ind : ElementTraits< Element< ET, EO > >::boundary_node_inds)
        nodes[bnd_node_ind] = n++;
    for (const auto& int_node_ind : ElementTraits< Element< ET, EO > >::internal_node_inds)
        nodes[int_node_ind] = n++;
    constexpr auto trans = [](auto inds) -> Point< 3 > {
        const auto [iz, iy, ix] = inds;
        if (ix == 0 and iy == 0 and iz == 0)
            return {0., 0., 0.};

        const auto     x     = static_cast< double >(ix);
        const auto     y     = static_cast< double >(iy);
        const auto     z     = static_cast< double >(iz);
        const auto     theta = std::atan2(std::sqrt(x * x + y * y), z);
        const auto     phi   = (ix != 0 or iy != 0) ? std::atan2(y, x) : 0.;
        constexpr auto r     = std::numbers::sqrt3;
        return {r * std::sin(theta) * std::cos(phi), r * std::sin(theta) * std::sin(phi), r * std::cos(theta)};
    };
    constexpr auto i3          = util::makeIotaArray< int, 3 >(-1);
    const auto     coord_range = std::views::cartesian_product(i3, i3, i3);
    std::ranges::transform(coord_range, data.vertices.begin(), trans);
    return retval;
}
#endif // L3STER_BENCHMARKS_COMMON_HPP
