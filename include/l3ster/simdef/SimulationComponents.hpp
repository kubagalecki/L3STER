#ifndef L3STER_SIMDEF_SIMULATIONCOMPONENTS_HPP
#define L3STER_SIMDEF_SIMULATIONCOMPONENTS_HPP

#include "l3ster/defs/Typedefs.h"
#include "l3ster/util/ConstexprRefStableCollection.hpp"

#include <string_view>
#include <tuple>

namespace lstr::def
{
template < typename... Kernels >
class SimulationComponents;
template < typename... Kernels >
class Simulation;

template < typename T >
struct Kernel
{
    std::string_view name;
    T                obj;
};
template < typename T >
Kernel(std::string_view, T&&) -> Kernel< std::remove_cvref_t< T > >;

template < typename... T >
class KernelSet
{
    friend class SimulationComponents< T... >;
    friend class Simulation< T... >;
    struct KernelToken
    {
        size_t index;
    };

public:
    constexpr KernelSet(const Kernel< T >&... objects) : m_objects{objects.obj...}, m_names{objects.name...}
    {
        auto names = m_names;
        std::ranges::sort(names);
        if (std::ranges::adjacent_find(names) != end(names))
            throw "Kernel names must be unique";
    }
    template < KernelToken token >
    [[nodiscard]] constexpr const auto& getObject() const
    {
        return std::get< token.index >(m_objects);
    }
    [[nodiscard]] constexpr auto getToken(std::string_view name) const
    {
        const auto it = std::ranges::find(m_names, name);
        if (it == end(m_names))
            throw "No kernel has this name";
        return KernelToken{static_cast< size_t >(std::distance(begin(m_names), it))};
    }

private:
    std::tuple< T... >                           m_objects;
    std::array< std::string_view, sizeof...(T) > m_names;
};
template < typename... T >
KernelSet(const Kernel< T >&...) -> KernelSet< T... >;

template < typename... Kernels >
class SimulationComponents
{
    friend class Simulation< Kernels... >;
    using kernel_token_t = typename KernelSet< Kernels... >::KernelToken;
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
        kernel_token_t                  kernel_token;
        ConstexprVector< const Field* > fields;
        ConstexprVector< d_id_t >       domains;
        q_o_t                           max_val_order, max_der_order;
    };
    struct DirichletBoundaryCondition
    {
        kernel_token_t                  kernel_token;
        ConstexprVector< const Field* > fields;
        ConstexprVector< d_id_t >       domains;
    };
    struct Operation
    {
        kernel_token_t kernel_token;
    };

public:
    constexpr const Field* addField(std::string_view name, std::uint8_t n_components, ConstexprVector< d_id_t > domains)
    {
        return std::addressof(m_fields.emplace(name, n_components, std::move(domains)));
    }
    constexpr const Value* addValue(std::string_view name, std::uint8_t n_components)
    {
        return std::addressof(m_values.emplace(name, n_components));
    }
    constexpr const Equation* addEquation(kernel_token_t                  kernel,
                                          ConstexprVector< const Field* > fields,
                                          ConstexprVector< d_id_t >       domains,
                                          q_o_t                           max_val_order,
                                          q_o_t                           max_der_order)
    {
        return std::addressof(
            m_equations.emplace(kernel, std::move(fields), std::move(domains), max_val_order, max_der_order));
    }
    constexpr const Equation* addBoundaryCondition(kernel_token_t                  kernel,
                                                   ConstexprVector< const Field* > fields,
                                                   ConstexprVector< d_id_t >       domains,
                                                   q_o_t                           max_val_order,
                                                   q_o_t                           max_der_order)
    {
        return std::addressof(
            m_boundary_conditions.emplace(kernel, std::move(fields), std::move(domains), max_val_order, max_der_order));
    }
    constexpr const DirichletBoundaryCondition* addDirichletBoundaryCondition(kernel_token_t                  kernel,
                                                                              ConstexprVector< const Field* > fields,
                                                                              ConstexprVector< d_id_t >       domains)
    {
        return std::addressof(m_dirichlet_conditions.emplace(kernel, std::move(fields), std::move(domains)));
    }
    constexpr const Operation* addDomainTransform(kernel_token_t kernel)
    {
        return std::addressof(m_domain_transforms.emplace(kernel));
    }
    constexpr const Operation* addBoundaryTransform(kernel_token_t kernel)
    {
        return std::addressof(m_boundary_transforms.emplace(kernel));
    }
    constexpr const Operation* addDomainReduction(kernel_token_t kernel)
    {
        return std::addressof(m_domain_reductions.emplace(kernel));
    }
    constexpr const Operation* addBoundaryReduction(kernel_token_t kernel)
    {
        return std::addressof(m_boundary_reductions.emplace(kernel));
    }

    constexpr const auto& getFields() const { return m_fields; }
    constexpr const auto& getValues() const { return m_values; }
    constexpr const auto& getEquations() const { return m_equations; }
    constexpr const auto& getBoundaryConditions() const { return m_boundary_conditions; }
    constexpr const auto& getDirichletConditions() const { return m_dirichlet_conditions; }
    constexpr const auto& getDomainTransforms() const { return m_domain_transforms; }
    constexpr const auto& getBoundaryTransforms() const { return m_boundary_transforms; }
    constexpr const auto& getDomainReductions() const { return m_domain_reductions; }
    constexpr const auto& getBoundaryReductions() const { return m_boundary_reductions; }

private:
    ConstexprRefStableCollection< Field >                      m_fields;
    ConstexprRefStableCollection< Value >                      m_values;
    ConstexprRefStableCollection< Equation >                   m_equations, m_boundary_conditions;
    ConstexprRefStableCollection< DirichletBoundaryCondition > m_dirichlet_conditions;
    ConstexprRefStableCollection< Operation > m_domain_transforms, m_boundary_transforms, m_domain_reductions,
        m_boundary_reductions;
};
} // namespace lstr::def
#endif // L3STER_SIMDEF_SIMULATIONCOMPONENTS_HPP
