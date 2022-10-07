#ifndef L3STER_SIMDEF_SIMULATION_HPP
#define L3STER_SIMDEF_SIMULATION_HPP

#include "l3ster/simdef/SimulationComponents.hpp"

namespace lstr::def
{
template < typename... Kernels >
class Simulation
{
    using kernel_set_t   = KernelSet< Kernels... >;
    using kernel_token_t = typename kernel_set_t::KernelToken;
    using components_t   = SimulationComponents< Kernels... >;
    using equation_t     = components_t::Equation;
    using dirichlet_bc_t = components_t::DirichletBoundaryCondition;

    struct Problem
    {
        ConstexprVector< const equation_t* >     equations;
        ConstexprVector< const equation_t* >     bcs;
        ConstexprVector< const dirichlet_bc_t* > dirichlet_bcs;
    };

public:
    constexpr Simulation(const Kernel< Kernels >&... kernels) : m_kernels{kernels...} {}

    constexpr auto        getKernelToken(std::string_view name) const { return m_kernels.getToken(name); }
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
    kernel_set_t                       m_kernels;
    SimulationComponents< Kernels... > m_components;
    ConstexprVector< Problem >         m_problems;
};
template < typename... Kernels >
Simulation(Kernel< Kernels >...) -> Simulation< Kernels... >;
} // namespace lstr::def
#endif // L3STER_SIMDEF_SIMULATION_HPP
