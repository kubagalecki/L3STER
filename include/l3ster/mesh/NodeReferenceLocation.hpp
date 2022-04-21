#ifndef L3STER_MESH_NODEREFERENCELOCATION_HPP
#define L3STER_MESH_NODEREFERENCELOCATION_HPP

#include "Point.hpp"
#include "l3ster/math/LobattoRuleAbsc.hpp"

namespace lstr
{
namespace detail
{
template < el_o_t O >
auto makeLineNodeLocations()
{
    std::array< Point< 1 >, O + 1 > ret_val;
    std::ranges::transform(
        getLobattoRuleAbsc< val_t, O + 1 >(), begin(ret_val), [](double x) { return Point< 1 >{x}; });
    return ret_val;
}

template < el_o_t O >
auto makeQuadNodeLocations()
{
    constexpr auto                                            nodes_per_edge = O + 1;
    std::array< Point< 2 >, nodes_per_edge * nodes_per_edge > ret_val;
    size_t                                                    i = 0;
    for (auto eta : getLobattoRuleAbsc< val_t, nodes_per_edge >())
        for (auto xi : getLobattoRuleAbsc< val_t, nodes_per_edge >())
            ret_val[i++] = Point< 2 >{xi, eta};
    return ret_val;
}

template < el_o_t O >
auto makeHexNodeLocations()
{
    constexpr auto                        nodes_per_edge = O + 1;
    constexpr auto                        total_nodes    = nodes_per_edge * nodes_per_edge * nodes_per_edge;
    std::array< Point< 3 >, total_nodes > ret_val;
    size_t                                i = 0;
    for (auto zeta : getLobattoRuleAbsc< val_t, nodes_per_edge >())
        for (auto eta : getLobattoRuleAbsc< val_t, nodes_per_edge >())
            for (auto xi : getLobattoRuleAbsc< val_t, nodes_per_edge >())
                ret_val[i++] = Point< 3 >{xi, eta, zeta};
    return ret_val;
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
