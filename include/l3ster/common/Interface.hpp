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
    dim_t  dimension{};
    size_t n_equations{};
    size_t n_unknowns{};
    size_t n_fields{};
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
    std::move_constructible< K > and std::invocable< std::add_const_t< std::decay_t< K > >,
                                                     const typename KernelInterface< params >::DomainInput&,
                                                     typename KernelInterface< params >::Result& >;

template < typename K, KernelParams params >
concept BoundaryKernel_c =
    std::move_constructible< K > and std::invocable< std::add_const_t< std::decay_t< K > >,
                                                     const typename KernelInterface< params >::BoundaryInput&,
                                                     typename KernelInterface< params >::Result& >;

template < typename K, KernelParams params >
concept ResidualDomainKernel_c =
    std::move_constructible< K > and std::invocable< std::add_const_t< std::decay_t< K > >,
                                                     const typename KernelInterface< params >::DomainInput&,
                                                     typename KernelInterface< params >::Rhs& >;

template < typename K, KernelParams params >
concept ResidualBoundaryKernel_c =
    std::move_constructible< K > and std::invocable< std::add_const_t< std::decay_t< K > >,
                                                     const typename KernelInterface< params >::BoundaryInput&,
                                                     typename KernelInterface< params >::Rhs& >;

template < typename Kernel, KernelParams params >
    requires DomainKernel_c< Kernel, params >
struct DomainKernel
{
    constexpr DomainKernel(Kernel kernel) : m_kernel{std::move(kernel)} {}

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
    requires BoundaryKernel_c< Kernel, params >
struct BoundaryKernel
{
    constexpr BoundaryKernel(Kernel kernel) : m_kernel{std::move(kernel)} {}

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
    requires ResidualDomainKernel_c< Kernel, params >
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
    requires ResidualBoundaryKernel_c< Kernel, params >
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
constexpr auto wrapDomainKernel(Kernel kernel, util::ConstexprValue< params > = {})
    requires DomainKernel_c< Kernel, params >
{
    return DomainKernel< std::remove_cvref_t< Kernel >, params >{std::move(kernel)};
}

template < KernelParams params, typename Kernel >
constexpr auto wrapBoundaryKernel(Kernel kernel, util::ConstexprValue< params > = {})
    requires BoundaryKernel_c< Kernel, params >
{
    return BoundaryKernel< std::remove_cvref_t< Kernel >, params >{std::move(kernel)};
}

template < KernelParams params, typename Kernel >
constexpr auto wrapResidualDomainKernel(Kernel kernel, util::ConstexprValue< params > = {})
    requires ResidualDomainKernel_c< Kernel, params >
{
    return ResidualDomainKernel< std::remove_cvref_t< Kernel >, params >{std::move(kernel)};
}

template < KernelParams params, typename Kernel >
constexpr auto wrapResidualBoundaryKernel(Kernel kernel, util::ConstexprValue< params > = {})
    requires ResidualBoundaryKernel_c< Kernel, params >
{
    return ResidualBoundaryKernel< std::remove_cvref_t< Kernel >, params >{std::move(kernel)};
}
} // namespace lstr
#endif // L3STER_COMMON_INTERFACE_HPP
