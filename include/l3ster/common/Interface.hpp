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
    size_t n_unknowns;
    size_t n_fields = 0;
};

template < KernelParams params >
struct KernelInterface
{
    using Operator = Eigen::Matrix< val_t, params.n_equations, params.n_unknowns >;
    using Rhs      = Eigen::Vector< val_t, params.n_equations >;
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
} // namespace detail

template < typename K, KernelParams params >
concept DomainKernel_c =
    ForwardConstructible_c< K > and std::invocable< std::add_const_t< std::decay_t< K > >,
                                                    const typename KernelInterface< params >::DomainInput&,
                                                    typename KernelInterface< params >::Result& >;

template < typename Kernel, KernelParams params >
struct DomainKernel
{
    constexpr DomainKernel(auto&& kernel, util::ConstexprValue< params >)
        requires DomainKernel_c< decltype(kernel), params >
        : m_kernel{std::forward< decltype(kernel) >(kernel)}
    {}

    auto operator()(const KernelInterface< params >::DomainInput& input) const -> KernelInterface< params >::Result
    {
        auto retval = detail::initKernelResult< params >();
        std::invoke(std::as_const(m_kernel), input, retval);
        return retval;
    }

private:
    Kernel m_kernel;
};
template < typename Kernel, KernelParams params >
DomainKernel(Kernel&&, util::ConstexprValue< params >) -> DomainKernel< std::decay_t< Kernel >, params >;

template < typename K, KernelParams params >
concept BoundaryKernel_c =
    ForwardConstructible_c< K > and std::invocable< std::add_const_t< std::decay_t< K > >,
                                                    const typename KernelInterface< params >::BoundaryInput&,
                                                    typename KernelInterface< params >::Result& >;

template < typename Kernel, KernelParams params >
struct BoundaryKernel
{
    constexpr BoundaryKernel(auto&& kernel, util::ConstexprValue< params >)
        requires BoundaryKernel_c< decltype(kernel), params >
        : m_kernel{std::forward< decltype(kernel) >(kernel)}
    {}

    auto operator()(const KernelInterface< params >::BoundaryInput& input) const -> KernelInterface< params >::Result
    {
        auto retval = detail::initKernelResult< params >();
        std::invoke(std::as_const(m_kernel), input, retval);
        return retval;
    }

private:
    Kernel m_kernel;
};
template < typename Kernel, KernelParams params >
BoundaryKernel(Kernel&&, util::ConstexprValue< params >) -> BoundaryKernel< std::decay_t< Kernel >, params >;

template < KernelParams params, typename Kernel >
constexpr auto wrapDomainKernel(Kernel&& kernel, util::ConstexprValue< params > params_ctwrpr = {})
    requires DomainKernel_c< Kernel, params >
{
    return DomainKernel{std::forward< Kernel >(kernel), params_ctwrpr};
}

template < KernelParams params, typename Kernel >
constexpr auto wrapBoundaryKernel(Kernel&& kernel, util::ConstexprValue< params > params_ctwrpr = {})
    requires BoundaryKernel_c< Kernel, params >
{
    return BoundaryKernel{std::forward< Kernel >(kernel), params_ctwrpr};
}
} // namespace lstr
#endif // L3STER_COMMON_INTERFACE_HPP
