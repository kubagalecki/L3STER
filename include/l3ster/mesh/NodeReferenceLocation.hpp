#ifndef L3STER_MESH_NODEREFERENCELOCATION_HPP
#define L3STER_MESH_NODEREFERENCELOCATION_HPP

#include "l3ster/math/LobattoRuleAbsc.hpp"
#include "l3ster/mesh/Point.hpp"

namespace lstr
{
namespace detail
{
template < el_o_t O >
auto makeLineNodeLocations()
{
    std::array< Point< 1 >, O + 1 > retval;
    std::ranges::transform(getLobattoRuleAbsc< val_t, O + 1 >(), begin(retval), [](val_t x) { return Point{x}; });
    return retval;
}

template < el_o_t O >
auto makeQuadNodeLocations()
{
    constexpr auto                                            nodes_per_edge = O + 1;
    const auto                                                absc = getLobattoRuleAbsc< val_t, nodes_per_edge >();
    std::array< Point< 2 >, nodes_per_edge * nodes_per_edge > retval;
    for (size_t i = 0; auto eta : absc)
        for (auto xi : absc)
            retval[i++] = Point{xi, eta};
    return retval;
}

template < el_o_t O >
auto makeHexNodeLocations()
{
    constexpr auto                        nodes_per_edge = O + 1;
    constexpr auto                        total_nodes    = nodes_per_edge * nodes_per_edge * nodes_per_edge;
    const auto                            absc           = getLobattoRuleAbsc< val_t, nodes_per_edge >();
    std::array< Point< 3 >, total_nodes > retval;
    for (size_t i = 0; auto zeta : absc)
        for (auto eta : absc)
            for (auto xi : absc)
                retval[i++] = Point{xi, eta, zeta};
    return retval;
}
} // namespace detail

template < ElementTypes T, el_o_t O >
const auto& getNodeLocations()
    requires(T == ElementTypes::Line or T == ElementTypes::Quad or T == ElementTypes::Hex)
{
    if constexpr (T == ElementTypes::Line)
    {
        static const auto value = detail::makeLineNodeLocations< O >();
        return value;
    }
    else if constexpr (T == ElementTypes::Quad)
    {
        static const auto value = detail::makeQuadNodeLocations< O >();
        return value;
    }
    else if constexpr (T == ElementTypes::Hex)
    {
        static const auto value = detail::makeHexNodeLocations< O >();
        return value;
    }
}
} // namespace lstr
#endif // L3STER_MESH_NODEREFERENCELOCATION_HPP
