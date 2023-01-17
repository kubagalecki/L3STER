#ifndef L3STER_SIMDEF_SIMULATIONCOMPONENTS_HPP
#define L3STER_SIMDEF_SIMULATIONCOMPONENTS_HPP

#include "l3ster/defs/Typedefs.h"
#include "l3ster/util/ConstexprRefStableCollection.hpp"

#include <string_view>
#include <tuple>

namespace lstr::def
{
template < std::copy_constructible K >
struct Kernel
{
    std::string_view name;
    K                kernel;
};
template < typename K >
Kernel(std::string_view, K&&) -> Kernel< std::decay_t< K > >;

template < std::copy_constructible... Kernels >
class KernelSet
{
public:
    static constexpr auto size = sizeof...(Kernels);

    constexpr KernelSet(const Kernel< Kernels >&... kernels) : m_kernels{kernels.kernel...}, m_names{kernels.name...}
    {
        auto names = m_names;
        std::ranges::sort(names);
        if (std::ranges::adjacent_find(names) != end(names))
            throw "Kernel names must be unique";
    }
    template < size_t index >
    [[nodiscard]] constexpr const auto& getKernel(std::integral_constant< size_t, index > = {}) const
        requires(index < size)
    {
        return std::get< index >(m_kernels);
    }
    [[nodiscard]] constexpr size_t getIndex(std::string_view name) const
    {
        const auto it = std::ranges::find(m_names, name);
        if (it == end(m_names))
            throw "The specified kernel is not present in this set";
        return static_cast< size_t >(std::distance(begin(m_names), it));
    }

private:
    std::tuple< Kernels... >             m_kernels;
    std::array< std::string_view, size > m_names;
};

template < std::copy_constructible... Kernels >
class Simulation;

template < std::copy_constructible... Kernels >
class SimulationComponents
{
    friend class Simulation< Kernels... >;
    struct Field
    {
        std::string_view          name;
        std::uint8_t              n_components;
        ConstexprVector< d_id_t > domains;
    };
    struct Value
    {
        std::string_view name;
        std::uint8_t     n_components;
    };
    struct Equation
    {
        size_t                          kernel_index;
        ConstexprVector< const Field* > fields;
        ConstexprVector< d_id_t >       domains;
        q_o_t                           max_val_order, max_der_order;
    };
    struct DirichletBoundaryCondition
    {
        size_t                          kernel_index;
        ConstexprVector< const Field* > fields;
        ConstexprVector< d_id_t >       domains;
    };
    struct Operation
    {
        size_t kernel_index;
    };
    struct Problem
    {
        ConstexprVector< const Equation* >                   equations;
        ConstexprVector< const Equation* >                   bcs;
        ConstexprVector< const DirichletBoundaryCondition* > dirichlet_bcs;
    };

public:
    constexpr SimulationComponents(const Kernel< Kernels >&... kernels) : m_kernels{kernels...} {}

    [[nodiscard]] constexpr const Field*
    defineField(std::string_view name, std::uint8_t n_components, ConstexprVector< d_id_t > domains)
    {
        return std::addressof(m_fields.emplace(name, n_components, std::move(domains)));
    }
    [[nodiscard]] constexpr const Value* defineValue(std::string_view name, std::uint8_t n_components)
    {
        return std::addressof(m_values.emplace(name, n_components));
    }
    [[nodiscard]] constexpr const Equation* defineEquation(std::string_view                kernel_name,
                                                           ConstexprVector< const Field* > fields,
                                                           ConstexprVector< d_id_t >       domains,
                                                           q_o_t                           max_val_order = 1,
                                                           q_o_t                           max_der_order = 0)
    {
        return std::addressof(m_equations.emplace(
            m_kernels.getIndex(kernel_name), std::move(fields), std::move(domains), max_val_order, max_der_order));
    }
    [[nodiscard]] constexpr const Equation* defineBoundaryCondition(std::string_view                kernel_name,
                                                                    ConstexprVector< const Field* > fields,
                                                                    ConstexprVector< d_id_t >       domains,
                                                                    q_o_t                           max_val_order = 1,
                                                                    q_o_t                           max_der_order = 0)
    {
        return std::addressof(m_boundary_conditions.emplace(
            m_kernels.getIndex(kernel_name), std::move(fields), std::move(domains), max_val_order, max_der_order));
    }
    [[nodiscard]] constexpr const DirichletBoundaryCondition* defineDirichletBoundaryCondition(
        std::string_view kernel_name, ConstexprVector< const Field* > fields, ConstexprVector< d_id_t > domains)
    {
        return std::addressof(
            m_dirichlet_conditions.emplace(m_kernels.getIndex(kernel_name), std::move(fields), std::move(domains)));
    }
    [[nodiscard]] constexpr const Operation* defineDomainTransform(std::string_view kernel_name)
    {
        return std::addressof(m_domain_transforms.emplace(m_kernels.getIndex(kernel_name)));
    }
    [[nodiscard]] constexpr const Operation* defineBoundaryTransform(std::string_view kernel_name)
    {
        return std::addressof(m_boundary_transforms.emplace(m_kernels.getIndex(kernel_name)));
    }
    [[nodiscard]] constexpr const Operation* defineValueTransform(std::string_view kernel_name)
    {
        return std::addressof(m_value_transforms.emplace(m_kernels.getIndex(kernel_name)));
    }
    [[nodiscard]] constexpr const Operation* defineDomainReduction(std::string_view kernel_name)
    {
        return std::addressof(m_domain_reductions.emplace(m_kernels.getIndex(kernel_name)));
    }
    [[nodiscard]] constexpr const Operation* defineBoundaryReduction(std::string_view kernel_name)
    {
        return std::addressof(m_boundary_reductions.emplace(m_kernels.getIndex(kernel_name)));
    }
    [[nodiscard]] constexpr const Problem*
    defineProblem(ConstexprVector< const Equation* >                   equations,
                  ConstexprVector< const Equation* >                   bcs,
                  ConstexprVector< const DirichletBoundaryCondition* > dirichlet_bcs)
    {
        return std::addressof(m_problems.emplace(std::move(equations), std::move(bcs), std::move(dirichlet_bcs)));
    }

    [[nodiscard]] constexpr const auto& getFields() const { return m_fields; }
    [[nodiscard]] constexpr const auto& getValues() const { return m_values; }
    [[nodiscard]] constexpr const auto& getEquations() const { return m_equations; }
    [[nodiscard]] constexpr const auto& getBoundaryConditions() const { return m_boundary_conditions; }
    [[nodiscard]] constexpr const auto& getDirichletConditions() const { return m_dirichlet_conditions; }
    [[nodiscard]] constexpr const auto& getDomainTransforms() const { return m_domain_transforms; }
    [[nodiscard]] constexpr const auto& getBoundaryTransforms() const { return m_boundary_transforms; }
    [[nodiscard]] constexpr const auto& getValueTransforms() const { return m_value_transforms; }
    [[nodiscard]] constexpr const auto& getDomainReductions() const { return m_domain_reductions; }
    [[nodiscard]] constexpr const auto& getBoundaryReductions() const { return m_boundary_reductions; }
    [[nodiscard]] constexpr const auto& getProblems() const { return m_problems; }

private:
    KernelSet< Kernels... >                                    m_kernels;
    ConstexprRefStableCollection< Field >                      m_fields;
    ConstexprRefStableCollection< Value >                      m_values;
    ConstexprRefStableCollection< Equation >                   m_equations, m_boundary_conditions;
    ConstexprRefStableCollection< DirichletBoundaryCondition > m_dirichlet_conditions;
    ConstexprRefStableCollection< Operation > m_domain_transforms, m_boundary_transforms, m_value_transforms,
        m_domain_reductions, m_boundary_reductions;
    ConstexprRefStableCollection< Problem > m_problems;
};
} // namespace lstr::def
#endif // L3STER_SIMDEF_SIMULATIONCOMPONENTS_HPP
