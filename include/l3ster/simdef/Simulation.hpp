#ifndef L3STER_SIMDEF_SIMULATION_HPP
#define L3STER_SIMDEF_SIMULATION_HPP

#include "l3ster/simdef/SimulationComponents.hpp"

namespace lstr::def
{
template < std::copy_constructible... Kernels >
class Simulation
{
    using equation_t     = SimulationComponents< Kernels... >::Equation;
    using dirichlet_bc_t = SimulationComponents< Kernels... >::DirichletBoundaryCondition;

    struct Problem
    {
        ConstexprVector< const equation_t* >     equations;
        ConstexprVector< const equation_t* >     bcs;
        ConstexprVector< const dirichlet_bc_t* > dirichlet_bcs;
    };

public:
    constexpr Simulation(const Kernel< Kernels >&... kernels)
        : m_components{std::forward< decltype(kernels) >(kernels)...}
    {}

    constexpr auto&       components() { return m_components; }
    constexpr const auto& components() const { return m_components; }

    constexpr const Problem* defineProblem(ConstexprVector< const equation_t* >     equations,
                                           ConstexprVector< const equation_t* >     bcs,
                                           ConstexprVector< const dirichlet_bc_t* > dirichlet_bcs)
    {
        return std::addressof(m_problems.emplaceBack(std::move(equations), std::move(bcs), std::move(dirichlet_bcs)));
    }

    constexpr const auto& getProblems() const { return m_problems; }

private:
    SimulationComponents< Kernels... > m_components;
    ConstexprVector< Problem >         m_problems;
};
template < std::copy_constructible... Kernels >
Simulation(const Kernel< Kernels >&...) -> Simulation< Kernels... >;
} // namespace lstr::def
#endif // L3STER_SIMDEF_SIMULATION_HPP
