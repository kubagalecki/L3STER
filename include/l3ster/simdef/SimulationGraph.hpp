#ifndef L3STER_SIMDEF_SIMULATIONGRAPH_HPP
#define L3STER_SIMDEF_SIMULATIONGRAPH_HPP

#include "l3ster/simdef/SimulationComponents.hpp"

namespace lstr::def
{
template < std::copy_constructible... Kernels >
class SimulationGraph
{};
} // namespace lstr::def
#endif // L3STER_SIMDEF_SIMULATIONGRAPH_HPP
