#ifndef L3STER_SIMSTRUCTURE_SIMULATIONDEF_HPP
#define L3STER_SIMSTRUCTURE_SIMULATIONDEF_HPP

#include "simstructure/MeshDef.hpp"
#include "simstructure/ProblemDef.hpp"

namespace lstr::def
{
template < Problem_c... P >
struct Simulation
{
    constexpr Simulation(const ProblemSet< P... >& problems_, const Mesh& mesh_)
        : problems{problems_.problems}, mesh{mesh_}
    {}

    std::tuple< P... > problems;
    Mesh               mesh;
};

template < Problem_c P >
Simulation(P, Mesh) -> Simulation< P >;
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_SIMULATIONDEF_HPP
