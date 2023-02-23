#ifndef L3STER_POST_INTEGRAL_HPP
#define L3STER_POST_INTEGRAL_HPP

#include "l3ster/assembly/AssembleGlobalSystem.hpp"
#include "l3ster/quad/EvalQuadrature.hpp"

namespace lstr
{
namespace detail
{
template < typename T, dim_t dim, size_t n_fields >
concept IntegralKernel_c = requires(T                                                int_kernel,
                                    std::array< val_t, n_fields >                    node_vals,
                                    std::array< std::array< val_t, n_fields >, dim > node_ders,
                                    SpaceTimePoint                                   point) {
                               {
                                   std::invoke(int_kernel, node_vals, node_ders, point)
                               } noexcept -> EigenVector_c;
                           };

template < typename T, dim_t dim, size_t n_fields >
concept BoundaryIntegralKernel_c = requires(T                                                int_kernel,
                                            std::array< val_t, n_fields >                    node_vals,
                                            std::array< std::array< val_t, n_fields >, dim > node_ders,
                                            SpaceTimePoint                                   point,
                                            Eigen::Vector< val_t, dim >                      normal) {
                                       {
                                           std::invoke(int_kernel, node_vals, node_ders, point, normal)
                                       } noexcept -> EigenVector_c;
                                   };

template < typename IntKernel, dim_t dim, size_t n_fields >
    requires IntegralKernel_c< IntKernel, dim, n_fields > or BoundaryIntegralKernel_c< IntKernel, dim, n_fields >
struct integral_kernel_eval_result
{};
template < typename IntKernel, dim_t dim, size_t n_fields >
    requires IntegralKernel_c< IntKernel, dim, n_fields >
struct integral_kernel_eval_result< IntKernel, dim, n_fields >
{
    using type = std::invoke_result_t< IntKernel,
                                       std::array< val_t, n_fields >,
                                       std::array< std::array< val_t, n_fields >, dim >,
                                       SpaceTimePoint >;
};
template < typename IntKernel, dim_t dim, size_t n_fields >
    requires BoundaryIntegralKernel_c< IntKernel, dim, n_fields >
struct integral_kernel_eval_result< IntKernel, dim, n_fields >
{
    using type = std::invoke_result_t< IntKernel,
                                       std::array< val_t, n_fields >,
                                       std::array< std::array< val_t, n_fields >, dim >,
                                       SpaceTimePoint,
                                       Eigen::Vector< val_t, dim > >;
};
template < typename IntKernel, dim_t dim, size_t n_fields >
    requires IntegralKernel_c< IntKernel, dim, n_fields > or BoundaryIntegralKernel_c< IntKernel, dim, n_fields >
using integral_kernel_eval_result_t = typename integral_kernel_eval_result< IntKernel, dim, n_fields >::type;
template < typename IntKernel, dim_t dim, size_t n_fields >
    requires IntegralKernel_c< IntKernel, dim, n_fields > or BoundaryIntegralKernel_c< IntKernel, dim, n_fields >
inline constexpr auto deduce_n_integral_components =
    integral_kernel_eval_result_t< IntKernel, dim, n_fields >::RowsAtCompileTime;
template < typename IntKernel, size_t n_fields >
inline constexpr auto n_integral_components = std::invoke(
    []< typename... TypeOrderPair >(TypePack< TypeOrderPair... >) {
        constexpr auto invalid_nic = std::numeric_limits< int >::max();
        constexpr auto deduce_nc   = []< ElementTypes ET, el_o_t EO >(ValuePack< ET, EO >) -> int {
            constexpr auto dim = Element< ET, EO >::native_dim;
            if constexpr (IntegralKernel_c< IntKernel, dim, n_fields > or
                          BoundaryIntegralKernel_c< IntKernel, dim, n_fields >)
                return deduce_n_integral_components< IntKernel, dim, n_fields >;
            else
                return invalid_nic;
        };
        constexpr auto int_comps_for_els = std::array< int, sizeof...(TypeOrderPair) >{deduce_nc(TypeOrderPair{})...};
        constexpr auto is_valid_nic      = [](int nic) {
            return nic != invalid_nic;
        };
        static_assert(std::ranges::find_if(int_comps_for_els, is_valid_nic) != end(int_comps_for_els));
        constexpr auto n_int_comps = *std::ranges::find_if(int_comps_for_els, is_valid_nic);
        static_assert(
            std::ranges::all_of(int_comps_for_els, [](int nic) { return nic == n_int_comps or nic == invalid_nic; }));
        return n_int_comps;
    },
    type_order_combinations{});

template < typename Kernel, size_t n_fields >
struct PotentiallyValidIntegralKernelDeductionHelper
{
    template < ElementTypes T, el_o_t O >
    struct DeductionHelperDomain
    {
        static constexpr bool value = IntegralKernel_c< Kernel, Element< T, O >::native_dim, n_fields >;
    };
    static constexpr bool domain = assert_any_element< DeductionHelperDomain >;

    template < ElementTypes T, el_o_t O >
    struct DeductionHelperBoundary
    {
        static constexpr bool value = BoundaryIntegralKernel_c< Kernel, Element< T, O >::native_dim, n_fields >;
    };
    static constexpr bool boundary = assert_any_element< DeductionHelperBoundary >;
};
template < typename Kernel, size_t n_fields >
concept PotentiallyValidIntegralKernel_c = PotentiallyValidIntegralKernelDeductionHelper< Kernel, n_fields >::domain;
template < typename Kernel, size_t n_fields >
concept PotentiallyValidBoundaryIntegralKernel_c =
    PotentiallyValidIntegralKernelDeductionHelper< Kernel, n_fields >::boundary;

template < ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto evalElementIntegral(auto&&                                                                    kernel,
                         const Element< ET, EO >&                                                  element,
                         const EigenRowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >& node_vals,
                         const ReferenceBasisAtQuadrature< ET, EO, QL >&                           basis_at_qps,
                         val_t                                                                     time)
    requires IntegralKernel_c< decltype(kernel), Element< ET, EO >::native_dim, n_fields >
{
    const auto jacobi_mat_generator = getNatJacobiMatGenerator(element);
    using result_t = integral_kernel_eval_result_t< decltype(kernel), Element< ET, EO >::native_dim, n_fields >;
    const auto compute_value_at_qp = [&](ptrdiff_t qp_ind, auto ref_coords) noexcept -> result_t {
        const auto jacobi_mat    = jacobi_mat_generator(ref_coords);
        const auto phys_coords   = mapToPhysicalSpace(element, ref_coords);
        const auto field_vals    = computeFieldVals(basis_at_qps.basis.values[qp_ind], node_vals);
        const auto field_ders    = computeFieldDers(basis_at_qps.basis.derivatives[qp_ind], node_vals);
        const auto kernel_result = std::invoke(kernel, field_vals, field_ders, SpaceTimePoint{phys_coords, time});
        return jacobi_mat.determinant() * kernel_result;
    };
    return evalQuadrature(compute_value_at_qp, basis_at_qps.quadrature, result_t{result_t::Zero()});
}

template < ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto evalElementBoundaryIntegral(auto&&                                                                    kernel,
                                 const BoundaryElementView< ET, EO >&                                      el_view,
                                 const EigenRowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >& node_vals,
                                 const ReferenceBasisAtQuadrature< ET, EO, QL >&                           basis_at_qps,
                                 val_t                                                                     time)
    requires BoundaryIntegralKernel_c< decltype(kernel), Element< ET, EO >::native_dim, n_fields >
{
    const auto jacobi_mat_generator = getNatJacobiMatGenerator(*el_view);
    using result_t = integral_kernel_eval_result_t< decltype(kernel), Element< ET, EO >::native_dim, n_fields >;
    const auto compute_value_at_qp = [&](ptrdiff_t qp_ind, auto ref_coords) noexcept -> result_t {
        const auto jacobi_mat  = jacobi_mat_generator(ref_coords);
        const auto phys_coords = mapToPhysicalSpace(*el_view, ref_coords);
        const auto normal      = computeBoundaryNormal(el_view, jacobi_mat);
        const auto field_vals  = computeFieldVals(basis_at_qps.basis.values[qp_ind], node_vals);
        const auto field_ders  = computeFieldDers(basis_at_qps.basis.derivatives[qp_ind], node_vals);
        const auto ker_res     = std::invoke(kernel, field_vals, field_ders, SpaceTimePoint{phys_coords, time}, normal);
        const auto bound_jac   = computeBoundaryIntegralJacobian(el_view, jacobi_mat);
        return bound_jac * ker_res;
    };
    return evalQuadrature(compute_value_at_qp, basis_at_qps.quadrature, result_t{result_t::Zero()});
}

template < BasisTypes BT, QuadratureTypes QT, q_o_t QO >
auto evalLocalIntegral(auto&&                  kernel,
                       const MeshPartition&    mesh,
                       DomainIdRange_c auto&&  domain_ids,
                       FieldValGetter_c auto&& field_val_getter,
                       val_t                   time)
    requires PotentiallyValidIntegralKernel_c< decltype(kernel), deduce_n_fields< decltype(field_val_getter) > >
{
    constexpr auto n_fields     = deduce_n_fields< decltype(field_val_getter) >;
    constexpr auto n_components = n_integral_components< decltype(kernel), n_fields >;
    using integral_t            = Eigen::Vector< val_t, n_components >;
    const auto reduce_element   = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) -> integral_t {
        constexpr auto el_dim = Element< ET, EO >::native_dim;
        if constexpr (IntegralKernel_c< decltype(kernel), el_dim, n_fields >)
        {
            const auto  field_vals = field_val_getter(element.getNodes());
            const auto& qbv        = getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
            return evalElementIntegral(kernel, element, field_vals, qbv, time);
        }
        else
        {
            std::cerr
                << "Attempting to integrate over an element for which the integration kernel is invalid. Please check "
                     "the kernel was defined correctly, and that you are integrating over the correct domain (e.g. that "
                     "you're not trying to evaluate a 2D kernel in a 3D domain). The program will now terminate.\n";
            std::terminate();
        }
    };
    return mesh.transformReduce(integral_t{integral_t::Zero()},
                                std::plus<>{},
                                reduce_element,
                                std::forward< decltype(domain_ids) >(domain_ids),
                                std::execution::par);
}

template < BasisTypes BT, QuadratureTypes QT, q_o_t QO >
auto evalLocalBoundaryIntegral(auto&&                  kernel,
                               const BoundaryView&     boundary,
                               FieldValGetter_c auto&& field_val_getter,
                               val_t                   time)
    requires PotentiallyValidBoundaryIntegralKernel_c< decltype(kernel), deduce_n_fields< decltype(field_val_getter) > >
{
    constexpr auto n_fields     = deduce_n_fields< decltype(field_val_getter) >;
    constexpr auto n_components = n_integral_components< decltype(kernel), n_fields >;
    using integral_t            = Eigen::Vector< val_t, n_components >;
    const auto reduce_element =
        [&]< ElementTypes ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) -> integral_t {
        constexpr auto el_dim = Element< ET, EO >::native_dim;
        if constexpr (BoundaryIntegralKernel_c< decltype(kernel), el_dim, n_fields >)
        {
            const auto  field_vals = field_val_getter(el_view->getNodes());
            const auto& qbv        = getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
            return evalElementBoundaryIntegral(kernel, el_view, field_vals, qbv, time);
        }
        else
        {
            std::cerr << "Attempting to integrate over an element boundary for which the integration kernel is "
                         "invalid. Please check the kernel was defined correctly, and that you are integrating over "
                         "the correct boundary (e.g. that you're not trying to evaluate a 2D kernel in a 3D domain). "
                         "The program will now terminate.\n";
            std::terminate();
        }
    };
    return boundary.reduce(integral_t{integral_t::Zero()}, reduce_element, std::plus<>{}, std::execution::par);
}
} // namespace detail

template < BasisTypes BT, QuadratureTypes QT, q_o_t QO >
auto evalIntegral(const MpiComm&                  comm,
                  auto&&                          kernel,
                  const MeshPartition&            mesh,
                  detail::DomainIdRange_c auto&&  domain_ids,
                  detail::FieldValGetter_c auto&& field_val_getter,
                  val_t                           time = 0.)
{
    const EigenVector_c auto local_integral =
        detail::evalLocalIntegral< BT, QT, QO >(std::forward< decltype(kernel) >(kernel),
                                                mesh,
                                                std::forward< decltype(domain_ids) >(domain_ids),
                                                std::forward< decltype(field_val_getter) >(field_val_getter),
                                                time);
    using integral_t = std::remove_const_t< decltype(local_integral) >;
    integral_t global_integral;
    comm.allReduce(
        std::views::counted(local_integral.data(), integral_t::RowsAtCompileTime), global_integral.data(), MPI_SUM);
    return global_integral;
}

template < BasisTypes BT, QuadratureTypes QT, q_o_t QO >
auto evalBoundaryIntegral(const MpiComm&                  comm,
                          auto&&                          kernel,
                          const BoundaryView&             boundary,
                          detail::FieldValGetter_c auto&& field_val_getter,
                          val_t                           time = 0.)
{
    const EigenVector_c auto local_integral =
        detail::evalLocalBoundaryIntegral< BT, QT, QO >(std::forward< decltype(kernel) >(kernel),
                                                        boundary,
                                                        std::forward< decltype(field_val_getter) >(field_val_getter),
                                                        time);
    using integral_t = std::remove_const_t< decltype(local_integral) >;
    integral_t global_integral;
    comm.allReduce(
        std::views::counted(local_integral.data(), integral_t::RowsAtCompileTime), global_integral.data(), MPI_SUM);
    return global_integral;
}
} // namespace lstr
#endif // L3STER_POST_INTEGRAL_HPP
