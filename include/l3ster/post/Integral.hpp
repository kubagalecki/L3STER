#ifndef L3STER_POST_INTEGRAL_HPP
#define L3STER_POST_INTEGRAL_HPP

#include "l3ster/algsys/AssembleGlobalSystem.hpp"
#include "l3ster/quad/EvalQuadrature.hpp"

namespace lstr
{
namespace post
{
template < typename Kernel, KernelParams params, mesh::ElementType ET, el_o_t EO, q_l_t QL >
auto evalElementIntegral(
    const ResidualDomainKernel< Kernel, params >&                                                  kernel,
    const mesh::Element< ET, EO >&                                                                 element,
    const util::eigen::RowMajorMatrix< val_t, mesh::Element< ET, EO >::n_nodes, params.n_fields >& node_vals,
    const basis::ReferenceBasisAtQuadrature< ET, EO, QL >&                                         basis_at_qps,
    val_t time) -> KernelInterface< params >::Rhs
{
    const auto jacobi_gen          = map::getNatJacobiMatGenerator(element.getData());
    const auto compute_value_at_qp = [&](ptrdiff_t qp_ind, const auto& ref_coords) -> KernelInterface< params >::Rhs {
        const auto& basis_vals           = basis_at_qps.basis.values[qp_ind];
        const auto& ref_ders             = basis_at_qps.basis.derivatives[qp_ind];
        const auto [phys_ders, jacobian] = map::mapDomain< ET, EO >(jacobi_gen, ref_coords, ref_ders);
        const auto kernel_result =
            algsys::evalKernel(kernel, ref_coords, basis_vals, phys_ders, node_vals, element.getData(), time);
        return jacobian * kernel_result;
    };
    return evalQuadrature(compute_value_at_qp, basis_at_qps.quadrature, detail::initResidualKernelResult< params >());
}

template < typename Kernel, KernelParams params, mesh::ElementType ET, el_o_t EO, q_l_t QL >
auto evalElementBoundaryIntegral(
    const ResidualBoundaryKernel< Kernel, params >&                                                kernel,
    const mesh::BoundaryElementView< ET, EO >&                                                     el_view,
    const util::eigen::RowMajorMatrix< val_t, mesh::Element< ET, EO >::n_nodes, params.n_fields >& node_vals,
    const basis::ReferenceBasisAtQuadrature< ET, EO, QL >&                                         basis_at_qps,
    val_t time) -> KernelInterface< params >::Rhs
{
    const auto jacobi_gen          = map::getNatJacobiMatGenerator(el_view->getData());
    const auto compute_value_at_qp = [&](ptrdiff_t qp_ind, const auto& ref_coords) -> KernelInterface< params >::Rhs {
        const auto& basis_vals                   = basis_at_qps.basis.values[qp_ind];
        const auto& ref_ders                     = basis_at_qps.basis.derivatives[qp_ind];
        const auto  side                         = el_view.getSide();
        const auto [phys_ders, jacobian, normal] = map::mapBoundary< ET, EO >(jacobi_gen, ref_coords, ref_ders, side);
        const auto kernel_result =
            algsys::evalKernel(kernel, ref_coords, basis_vals, phys_ders, node_vals, el_view->getData(), time, normal);
        return jacobian * kernel_result;
    };
    return evalQuadrature(compute_value_at_qp, basis_at_qps.quadrature, detail::initResidualKernelResult< params >());
}

template < typename Kernel, KernelParams params, el_o_t... orders, AssemblyOptions options >
auto evalLocalIntegral(const ResidualDomainKernel< Kernel, params >&               kernel,
                       const mesh::MeshPartition< orders... >&                     mesh,
                       const util::ArrayOwner< d_id_t >&                           domain_ids,
                       const SolutionManager::FieldValueGetter< params.n_fields >& field_val_getter,
                       util::ConstexprValue< options >,
                       val_t time) -> KernelInterface< params >::Rhs
{
    const auto reduce_element = [&]< mesh::ElementType ET, el_o_t EO >(
                                    const mesh::Element< ET, EO >& element) -> KernelInterface< params >::Rhs {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            constexpr auto BT         = options.basis_type;
            constexpr auto QT         = options.quad_type;
            constexpr auto QO         = options.order(EO);
            const auto     field_vals = field_val_getter.getGloballyIndexed(element.getNodes());
            const auto&    qbv        = basis::getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
            return evalElementIntegral(kernel, element, field_vals, qbv, time);
        }
        else
            return detail::initResidualKernelResult< params >();
    };
    const auto zero = detail::initResidualKernelResult< params >();
    return mesh.transformReduce(domain_ids, zero, reduce_element, std::plus{}, std::execution::par);
}

template < typename Kernel, KernelParams params, el_o_t... orders, AssemblyOptions options >
auto evalLocalIntegral(const ResidualBoundaryKernel< Kernel, params >&             kernel,
                       const mesh::MeshPartition< orders... >&                     mesh,
                       const util::ArrayOwner< d_id_t >&                           boundary_ids,
                       const SolutionManager::FieldValueGetter< params.n_fields >& field_val_getter,
                       util::ConstexprValue< options >,
                       val_t time) -> KernelInterface< params >::Rhs
{
    const auto reduce_element =
        [&]< mesh::ElementType ET, el_o_t EO >(
            const mesh::BoundaryElementView< ET, EO >& el_view) -> KernelInterface< params >::Rhs {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            constexpr auto BT         = options.basis_type;
            constexpr auto QT         = options.quad_type;
            constexpr auto QO         = options.order(EO);
            const auto     field_vals = field_val_getter.getGloballyIndexed(el_view->getNodes());
            const auto&    qbv = basis::getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(el_view.getSide());
            return evalElementBoundaryIntegral(kernel, el_view, field_vals, qbv, time);
        }
        else
            return detail::initResidualKernelResult< params >();
    };
    const auto zero = detail::initResidualKernelResult< params >();
    return mesh.transformReduceBoundaries(boundary_ids, zero, reduce_element, std::plus{}, std::execution::par);
}
} // namespace post

template < ResidualKernel_c Kernel, el_o_t... orders, AssemblyOptions opts = {} >
auto computeIntegral(const MpiComm&                                                          comm,
                     const Kernel&                                                           kernel,
                     const mesh::MeshPartition< orders... >&                                 mesh,
                     const util::ArrayOwner< d_id_t >&                                       domain_ids,
                     const SolutionManager::FieldValueGetter< Kernel::parameters.n_fields >& field_val_getter = {},
                     util::ConstexprValue< opts >                                            opts_ctwrpr      = {},
                     val_t                                                                   time             = 0.)
{
    const auto local_integral  = post::evalLocalIntegral(kernel, mesh, domain_ids, field_val_getter, opts_ctwrpr, time);
    auto       global_integral = detail::initResidualKernelResult< Kernel::parameters >();
    auto       comm_view       = std::views::counted(local_integral.data(), Kernel::parameters.n_equations);
    comm.allReduce(std::move(comm_view), global_integral.data(), MPI_SUM);
    return global_integral;
}
} // namespace lstr
#endif // L3STER_POST_INTEGRAL_HPP
