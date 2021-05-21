#ifndef L3STER_MESH_NODEREFERENCELOCATION_HPP
#define L3STER_MESH_NODEREFERENCELOCATION_HPP

#include "math/LobattoRuleAbsc.hpp"
#include "mesh/Point.hpp"

namespace lstr
{
namespace detail
{
template < el_o_t O >
auto getLineNodeLocations()
{
    std::array< Point< 1 >, O + 1 > retval;
    std::ranges::transform(lobatto_rule_absc< val_t, O + 1 >, begin(retval), [](double x) { return Point< 1 >{x}; });
    return retval;
}

template < el_o_t O >
auto getQuadNodeLocations()
{
    constexpr auto                                            nodes_per_edge = O + 1;
    std::array< Point< 2 >, nodes_per_edge * nodes_per_edge > retval;
    size_t                                                    i = 0;
    for (auto eta : lobatto_rule_absc< val_t, nodes_per_edge >)
        for (auto xi : lobatto_rule_absc< val_t, nodes_per_edge >)
            retval[i++] = Point< 2 >{xi, eta};
    return retval;
}

template < el_o_t O >
auto getHexNodeLocations()
{
    constexpr auto                        nodes_per_edge = O + 1;
    constexpr auto                        total_nodes    = nodes_per_edge * nodes_per_edge * nodes_per_edge;
    std::array< Point< 3 >, total_nodes > retval;
    size_t                                i = 0;
    for (auto zeta : lobatto_rule_absc< val_t, nodes_per_edge >)
        for (auto eta : lobatto_rule_absc< val_t, nodes_per_edge >)
            for (auto xi : lobatto_rule_absc< val_t, nodes_per_edge >)
                retval[i++] = Point< 3 >{xi, eta, zeta};
    return retval;
}

template < ElementTypes T, el_o_t O >
auto getNodeLocations()
{
    if constexpr (T == ElementTypes::Line)
        return getLineNodeLocations< O >();
    else if constexpr (T == ElementTypes::Quad)
        return getQuadNodeLocations< O >();
    else if constexpr (T == ElementTypes::Hex)
        return getHexNodeLocations< O >();
}
} // namespace detail

template < ElementTypes T, el_o_t O >
requires(T == ElementTypes::Line or T == ElementTypes::Quad or T == ElementTypes::Hex) inline const
    auto node_reference_locations = detail::getNodeLocations< T, O >();
} // namespace lstr
#endif // L3STER_MESH_NODEREFERENCELOCATION_HPP
