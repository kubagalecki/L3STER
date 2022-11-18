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

template < typename IntKernel, ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto evalElementIntegral(IntKernel&&                                                                    int_kernel,
                         const Element< ET, EO >&                                                       element,
                         const EigenRowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >&      node_vals,
                         const ReferenceBasisAtQuadrature< ET, EO, QL, Element< ET, EO >::native_dim >& basis_at_qps,
                         val_t                                                                          time)
    requires detail::IntegralKernel_c< IntKernel, Element< ET, EO >::native_dim, n_fields >
{
    const auto jacobi_mat_generator = getNatJacobiMatGenerator(element);
    using result_t = detail::integral_kernel_eval_result_t< IntKernel, Element< ET, EO >::native_dim, n_fields >;
    const auto compute_value_at_qp = [&](ptrdiff_t qp_ind, auto ref_coords) noexcept -> result_t {
        const auto jacobi_mat    = jacobi_mat_generator(ref_coords);
        const auto phys_coords   = mapToPhysicalSpace(element, ref_coords);
        const auto field_vals    = detail::computeFieldVals(basis_at_qps.basis.values[qp_ind], node_vals);
        const auto field_ders    = detail::computeFieldDers(basis_at_qps.basis.derivatives[qp_ind], node_vals);
        const auto kernel_result = std::invoke(int_kernel, field_vals, field_ders, SpaceTimePoint{phys_coords, time});
        return jacobi_mat.determinant() * kernel_result;
    };
    return evalQuadrature(compute_value_at_qp, basis_at_qps.quadrature, result_t{result_t::Zero()});
}

template < typename IntKernel, ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto evalElementBoundaryIntegral(
    IntKernel&&                                                                    int_kernel,
    const BoundaryElementView< ET, EO >&                                           el_view,
    const EigenRowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >&      node_vals,
    const ReferenceBasisAtQuadrature< ET, EO, QL, Element< ET, EO >::native_dim >& basis_at_qps,
    val_t                                                                          time)
    requires detail::BoundaryIntegralKernel_c< IntKernel, Element< ET, EO >::native_dim, n_fields >
{
    const auto jacobi_mat_generator = getNatJacobiMatGenerator(*el_view);
    using result_t = detail::integral_kernel_eval_result_t< IntKernel, Element< ET, EO >::native_dim, n_fields >;
    const auto compute_value_at_qp = [&](ptrdiff_t qp_ind, auto ref_coords) noexcept -> result_t {
        const auto jacobi_mat  = jacobi_mat_generator(ref_coords);
        const auto phys_coords = mapToPhysicalSpace(*el_view, ref_coords);
        const auto normal      = computeBoundaryNormal(el_view, jacobi_mat);
        const auto field_vals  = detail::computeFieldVals(basis_at_qps.basis.values[qp_ind], node_vals);
        const auto field_ders  = detail::computeFieldDers(basis_at_qps.basis.derivatives[qp_ind], node_vals);
        const auto ker_res = std::invoke(int_kernel, field_vals, field_ders, SpaceTimePoint{phys_coords, time}, normal);
        return jacobi_mat.determinant() * ker_res;
    };
    return evalQuadrature(compute_value_at_qp, basis_at_qps.quadrature, result_t{result_t::Zero()});
}

template < BasisTypes      BT,
           QuadratureTypes QT,
           q_o_t           QO,
           typename IntKernel,
           detail::FieldValGetter_c FvalGetter,
           detail::DomainIdRange_c  R >
auto evalLocalIntegral(
    IntKernel&& int_kernel, const MeshPartition& mesh, R&& domain_ids, FvalGetter&& field_val_getter, val_t time)
{
    constexpr auto n_fields     = detail::deduce_n_fields< FvalGetter >;
    constexpr auto n_components = detail::n_integral_components< IntKernel, n_fields >;
    using integral_t            = Eigen::Vector< val_t, n_components >;
    const auto reduce_element   = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) -> integral_t {
        constexpr auto el_dim = Element< ET, EO >::native_dim;
        if constexpr (detail::IntegralKernel_c< IntKernel, el_dim, n_fields >)
        {
            const auto  field_vals = field_val_getter(element.getNodes());
            const auto& qbv        = getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
            return evalElementIntegral(int_kernel, element, field_vals, qbv, time);
        }
        else
            return integral_t::Zero(); // Note: this is logically unreachable, but we need a return statement
    };
    return mesh.reduce(integral_t{integral_t::Zero()},
                       reduce_element,
                       std::plus<>{},
                       std::forward< R >(domain_ids),
                       std::execution::par);
}

template < BasisTypes BT, QuadratureTypes QT, q_o_t QO, typename IntKernel, detail::FieldValGetter_c FvalGetter >
auto evalLocalBoundaryIntegral(IntKernel&&         int_kernel,
                               const BoundaryView& boundary,
                               FvalGetter&&        field_val_getter,
                               val_t               time)
{
    constexpr auto n_fields     = detail::deduce_n_fields< FvalGetter >;
    constexpr auto n_components = detail::n_integral_components< IntKernel, n_fields >;
    using integral_t            = Eigen::Vector< val_t, n_components >;
    const auto reduce_element =
        [&]< ElementTypes ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) -> integral_t {
        constexpr auto el_dim = Element< ET, EO >::native_dim;
        if constexpr (detail::BoundaryIntegralKernel_c< IntKernel, el_dim, n_fields >)
        {
            const auto  field_vals = field_val_getter(el_view->getNodes());
            const auto& qbv        = getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
            return evalElementBoundaryIntegral(int_kernel, el_view, field_vals, qbv, time);
        }
        else
            return integral_t::Zero(); // Note: this is logically unreachable, but we need a return statement
    };
    return boundary.reduce(integral_t{integral_t::Zero()}, reduce_element, std::plus<>{}, std::execution::par);
}
} // namespace detail

template < BasisTypes      BT,
           QuadratureTypes QT,
           q_o_t           QO,
           typename IntKernel,
           detail::FieldValGetter_c FvalGetter,
           detail::DomainIdRange_c  R >
auto evalIntegral(const MpiComm&       comm,
                  IntKernel&&          int_kernel,
                  const MeshPartition& mesh,
                  R&&                  domain_ids,
                  FvalGetter&&         field_val_getter,
                  val_t                time = 0.)
{
    const EigenVector_c auto local_integral =
        detail::evalLocalIntegral< BT, QT, QO >(std::forward< IntKernel >(int_kernel),
                                                mesh,
                                                std::forward< R >(domain_ids),
                                                std::forward< FvalGetter >(field_val_getter),
                                                time);
    using integral_t = std::remove_const_t< decltype(local_integral) >;
    integral_t global_integral;
    comm.allReduce(local_integral.data(), global_integral.data(), integral_t::RowsAtCompileTime, MPI_SUM);
    return global_integral;
}

template < BasisTypes BT, QuadratureTypes QT, q_o_t QO, typename IntKernel, detail::FieldValGetter_c FvalGetter >
auto evalBoundaryIntegral(const MpiComm&      comm,
                          IntKernel&&         int_kernel,
                          const BoundaryView& boundary,
                          FvalGetter&&        field_val_getter,
                          val_t               time = 0.)
{
    const EigenVector_c auto local_integral = detail::evalLocalBoundaryIntegral< BT, QT, QO >(
        std::forward< IntKernel >(int_kernel), boundary, std::forward< FvalGetter >(field_val_getter), time);
    using integral_t = std::remove_const_t< decltype(local_integral) >;
    integral_t global_integral;
    comm.allReduce(local_integral.data(), global_integral.data(), integral_t::RowsAtCompileTime, MPI_SUM);
    return global_integral;
}
} // namespace lstr
#endif // L3STER_POST_INTEGRAL_HPP
