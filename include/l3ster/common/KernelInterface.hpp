#ifndef L3STER_COMMON_INTERFACE_HPP
#define L3STER_COMMON_INTERFACE_HPP

#include "l3ster/util/Common.hpp"
#include "l3ster/util/EigenUtils.hpp"

#include <array>
#include <concepts>

namespace lstr
{
struct KernelParams
{
    dim_t  dimension;
    size_t n_equations;
    size_t n_unknowns = 1;
    size_t n_fields   = 0;
    size_t n_rhs      = 1;
};

template < KernelParams params >
struct KernelInterface
{
    using Operator = Eigen::Matrix< val_t, params.n_equations, params.n_unknowns >;
    using Rhs      = Eigen::Matrix< val_t, params.n_equations, params.n_rhs >;
    struct Result
    {
        std::array< Operator, params.dimension + 1 > operators;
        Rhs                                          rhs;
    };

    using FieldVals = std::array< val_t, params.n_fields >;
    using FieldDers = std::array< FieldVals, params.dimension >;
    using Normal    = Eigen::Vector< val_t, params.dimension >;

    struct DomainInput
    {
        FieldVals      field_vals;
        FieldDers      field_ders;
        SpaceTimePoint point;
    };
    struct BoundaryInput
    {
        FieldVals      field_vals;
        FieldDers      field_ders;
        SpaceTimePoint point;
        Normal         normal;
    };
};

namespace detail
{
template < KernelParams params >
auto initKernelResult() -> KernelInterface< params >::Result
{
    auto retval = typename KernelInterface< params >::Result{};
    for (auto& op : retval.operators)
        op.setZero();
    retval.rhs.setZero();
    return retval;
}

template < KernelParams params >
auto initResidualKernelResult() -> KernelInterface< params >::Rhs
{
    return KernelInterface< params >::Rhs::Zero();
}
} // namespace detail

template < typename K, KernelParams params >
concept DomainEquationKernel_c =
    std::move_constructible< K > and std::invocable< std::add_const_t< std::decay_t< K > >,
                                                     const typename KernelInterface< params >::DomainInput&,
                                                     typename KernelInterface< params >::Result& >;

template < typename K, KernelParams params >
concept BoundaryEquationKernel_c =
    std::move_constructible< K > and std::invocable< std::add_const_t< std::decay_t< K > >,
                                                     const typename KernelInterface< params >::BoundaryInput&,
                                                     typename KernelInterface< params >::Result& >;

template < typename K, KernelParams params >
concept DomainResidualKernel_c =
    std::move_constructible< K > and std::invocable< std::add_const_t< std::decay_t< K > >,
                                                     const typename KernelInterface< params >::DomainInput&,
                                                     typename KernelInterface< params >::Rhs& >;

template < typename K, KernelParams params >
concept BoundaryResidualKernel_c =
    std::move_constructible< K > and std::invocable< std::add_const_t< std::decay_t< K > >,
                                                     const typename KernelInterface< params >::BoundaryInput&,
                                                     typename KernelInterface< params >::Rhs& >;

template < typename Kernel, KernelParams params >
    requires DomainEquationKernel_c< Kernel, params >
struct DomainEquationKernel
{
    constexpr DomainEquationKernel(Kernel kernel) : m_kernel{std::move(kernel)} {}

    auto operator()(const KernelInterface< params >::DomainInput& input) const -> KernelInterface< params >::Result
    {
        auto retval = detail::initKernelResult< params >();
        std::invoke(m_kernel, input, retval);
        return retval;
    }

private:
    Kernel m_kernel;
};

template < typename Kernel, KernelParams params >
    requires BoundaryEquationKernel_c< Kernel, params >
struct BoundaryEquationKernel
{
    constexpr BoundaryEquationKernel(Kernel kernel) : m_kernel{std::move(kernel)} {}

    auto operator()(const KernelInterface< params >::BoundaryInput& input) const -> KernelInterface< params >::Result
    {
        auto retval = detail::initKernelResult< params >();
        std::invoke(m_kernel, input, retval);
        return retval;
    }

private:
    Kernel m_kernel;
};

template < typename Kernel, KernelParams params >
    requires DomainResidualKernel_c< Kernel, params >
struct ResidualDomainKernel
{
    constexpr ResidualDomainKernel(Kernel kernel) : m_kernel{std::move(kernel)} {}

    auto operator()(const KernelInterface< params >::DomainInput& input) const -> KernelInterface< params >::Rhs
    {
        auto retval = typename KernelInterface< params >::Rhs{};
        std::invoke(m_kernel, input, retval);
        return retval;
    }

private:
    Kernel m_kernel;
};

template < typename Kernel, KernelParams params >
    requires BoundaryResidualKernel_c< Kernel, params >
struct ResidualBoundaryKernel
{
    constexpr ResidualBoundaryKernel(Kernel kernel) : m_kernel{std::move(kernel)} {}

    auto operator()(const KernelInterface< params >::BoundaryInput& input) const -> KernelInterface< params >::Rhs
    {
        auto retval = typename KernelInterface< params >::Rhs{};
        std::invoke(m_kernel, input, retval);
        return retval;
    }

private:
    Kernel m_kernel;
};

template < KernelParams params, typename Kernel >
constexpr auto wrapDomainEquationKernel(Kernel kernel, util::ConstexprValue< params > = {})
    requires DomainEquationKernel_c< Kernel, params >
{
    return DomainEquationKernel< std::remove_cvref_t< Kernel >, params >{std::move(kernel)};
}

template < KernelParams params, typename Kernel >
constexpr auto wrapBoundaryEquationKernel(Kernel kernel, util::ConstexprValue< params > = {})
    requires BoundaryEquationKernel_c< Kernel, params >
{
    return BoundaryEquationKernel< std::remove_cvref_t< Kernel >, params >{std::move(kernel)};
}

template < KernelParams params, typename Kernel >
constexpr auto wrapDomainResidualKernel(Kernel kernel, util::ConstexprValue< params > = {})
    requires DomainResidualKernel_c< Kernel, params >
{
    return ResidualDomainKernel< std::remove_cvref_t< Kernel >, params >{std::move(kernel)};
}

template < KernelParams params, typename Kernel >
constexpr auto wrapBoundaryResidualKernel(Kernel kernel, util::ConstexprValue< params > = {})
    requires BoundaryResidualKernel_c< Kernel, params >
{
    return ResidualBoundaryKernel< std::remove_cvref_t< Kernel >, params >{std::move(kernel)};
}
} // namespace lstr
#endif // L3STER_COMMON_INTERFACE_HPP
