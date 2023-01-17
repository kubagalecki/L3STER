#ifndef L3STER_SIMDEF_SIMULATION_HPP
#define L3STER_SIMDEF_SIMULATION_HPP

#include "l3ster/simdef/SimulationGraph.hpp"

namespace lstr::def
{
template < std::copy_constructible... Kernels >
class Simulation
{
public:
    constexpr Simulation(const Kernel< Kernels >&... kernels)
        : m_components{std::forward< decltype(kernels) >(kernels)...}
    {}

    constexpr auto&       components() { return m_components; }
    constexpr const auto& components() const { return m_components; }
    constexpr auto&       graph() { return m_graph; }
    constexpr const auto& graph() const { return m_graph; }

private:
    SimulationComponents< Kernels... > m_components;
    SimulationGraph< Kernels... >      m_graph;
};
template < std::copy_constructible... Kernels >
Simulation(const Kernel< Kernels >&...) -> Simulation< Kernels... >;
} // namespace lstr::def
#endif // L3STER_SIMDEF_SIMULATION_HPP
