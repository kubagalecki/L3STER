#ifndef L3STER_ASSEMBLY_KERNELTRAITS_HPP
#define L3STER_ASSEMBLY_KERNELTRAITS_HPP

#include "l3ster/util/IncludeEigen.hpp"

namespace lstr::detail
{
template < std::size_t n_ders >
struct KernelFieldDependence
{
    std::size_t                       n_fields;
    std::array< std::size_t, n_ders > der_inds;
};

template < typename >
inline constexpr bool is_kernel_fdep = false;
template < std::size_t n_ders >
inline constexpr bool is_kernel_fdep< KernelFieldDependence< n_ders > > = true;
template < typename T >
concept KernelFieldDependence_c = is_kernel_fdep< T >;

template < typename T >
concept ValidKernelResult_c = pair< T > and array< typename T::first_type > and
                              EigenMatrix_c< typename T::first_type::value_type > and
                              EigenMatrix_c< typename T::second_type > and(T::second_type::ColsAtCompileTime == 1) and
                              (T::first_type::value_type::RowsAtCompileTime == T::second_type::RowsAtCompileTime);

template < typename Kernel, ElementTypes ET, el_o_t EO, auto field_dep >
concept FullKernel_c =
    requires(Kernel                                                                                      k,
             std::array< val_t, field_dep.n_fields >                                                     node_vals,
             std::array< std::array< val_t, field_dep.der_inds.size() >, Element< ET, EO >::native_dim > node_ders,
             Point< 3 >                                                                                  point)
{
    {
        std::invoke(k, node_vals, node_ders, point)
    }
    noexcept->ValidKernelResult_c;
};

template < typename Kernel, ElementTypes ET, el_o_t EO, auto field_dep >
concept SpaceIndependentKernel_c =
    (field_dep.n_fields > 0) and (field_dep.der_inds.size() <= field_dep.n_fields) and
    requires(Kernel                                                                                      k,
             std::array< val_t, field_dep.n_fields >                                                     node_vals,
             std::array< std::array< val_t, field_dep.der_inds.size() >, Element< ET, EO >::native_dim > node_ders)
{
    {
        std::invoke(k, node_vals, node_ders)
    }
    noexcept->ValidKernelResult_c;
};

template < typename Kernel, ElementTypes ET, el_o_t EO, auto field_dep >
concept DerivativeIndependentKernel_c = (field_dep.n_fields > 0) and (field_dep.der_inds.size() == 0) and
                                        requires(Kernel k, std::array< val_t, field_dep.n_fields > node_vals)
{
    {
        std::invoke(k, node_vals)
    }
    noexcept->ValidKernelResult_c;
};

template < typename Kernel, ElementTypes ET, el_o_t EO, auto field_dep >
concept ConstantKernel_c = (field_dep.n_fields == 0) and (field_dep.der_inds.size() == 0) and requires(Kernel k)
{
    {
        std::invoke(k)
    }
    noexcept->ValidKernelResult_c;
};

template < typename Kernel, ElementTypes ET, el_o_t EO, auto field_dep >
concept ValidKernel_c =
    FullKernel_c< Kernel, ET, EO, field_dep > or SpaceIndependentKernel_c< Kernel, ET, EO, field_dep > or
    DerivativeIndependentKernel_c< Kernel, ET, EO, field_dep > or ConstantKernel_c< Kernel, ET, EO, field_dep >;

template < typename Kernel, ElementTypes ET, el_o_t EO, auto field_dep >
struct KernelReturnTypeHelper;
template < typename Kernel, ElementTypes ET, el_o_t EO, auto field_dep >
requires FullKernel_c< Kernel, ET, EO, field_dep >
struct KernelReturnTypeHelper< Kernel, ET, EO, field_dep >
{
    using type = std::invoke_result_t<
        Kernel,
        std::array< val_t, field_dep.n_fields >,
        std::array< std::array< val_t, field_dep.der_inds.size() >, Element< ET, EO >::native_dim >,
        Point< 3 > >;
};
template < typename Kernel, ElementTypes ET, el_o_t EO, auto field_dep >
requires SpaceIndependentKernel_c< Kernel, ET, EO, field_dep >
struct KernelReturnTypeHelper< Kernel, ET, EO, field_dep >
{
    using type = std::invoke_result_t<
        Kernel,
        std::array< val_t, field_dep.n_fields >,
        std::array< std::array< val_t, field_dep.der_inds.size() >, Element< ET, EO >::native_dim > >;
};
template < typename Kernel, ElementTypes ET, el_o_t EO, auto field_dep >
requires DerivativeIndependentKernel_c< Kernel, ET, EO, field_dep >
struct KernelReturnTypeHelper< Kernel, ET, EO, field_dep >
{
    using type = std::invoke_result_t< Kernel, std::array< val_t, field_dep.n_fields > >;
};
template < typename Kernel, ElementTypes ET, el_o_t EO, auto field_dep >
requires ConstantKernel_c< Kernel, ET, EO, field_dep >
struct KernelReturnTypeHelper< Kernel, ET, EO, field_dep >
{
    using type = std::invoke_result_t< Kernel >;
};
template < typename Kernel, ElementTypes ET, el_o_t EO, auto field_dep >
using kernel_return_t = typename KernelReturnTypeHelper< Kernel, ET, EO, field_dep >::type;
template < typename Kernel, ElementTypes ET, el_o_t EO, auto field_dep >
inline constexpr std::size_t n_unknowns =
    kernel_return_t< Kernel, ET, EO, field_dep >::first_type::value_type::ColsAtCompileTime;
} // namespace lstr::detail
#endif // L3STER_ASSEMBLY_KERNELTRAITS_HPP
