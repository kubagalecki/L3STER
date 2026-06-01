#ifndef L3STER_POST_INTEGRAL_HPP
#define L3STER_POST_INTEGRAL_HPP

#include "l3ster/algsys/AssembleGlobalSystem.hpp"

namespace lstr
{
namespace post
{
template < typename Kernel, KernelParams params, size_t NB, size_t QL >
auto evalElementIntegral(const ResidualDomainKernel< Kernel, params >&                  kernel,
                         const map::TabulatedDomainMapping< NB, QL, params.dimension >& map_result,
                         const Eigen::Matrix< val_t, int{NB}, params.n_fields >&        node_vals,
                         std::span< const val_t, QL >                                   quad_weights,
                         val_t time) -> KernelInterface< params >::Rhs
{
    const auto& [J, vals, ders, points] = map_result;
    const auto fields                   = map::FieldValuesAtPoints{*vals, ders, node_vals};
    auto       retval                   = lstr::detail::initResidualKernelResult< params >();
    for (size_t p = 0; p != QL; ++p)
    {
        const auto point              = SpaceTimePoint{points[p], time};
        const auto [field_v, field_d] = fields.get(p);
        const auto wgt                = quad_weights[p] * J[p];
        retval += wgt * kernel({field_v, field_d, point});
    }
    return retval;
}

template < typename Kernel, KernelParams params, size_t NB, size_t QL >
auto evalElementBoundaryIntegral(const ResidualBoundaryKernel< Kernel, params >&                  kernel,
                                 const map::TabulatedBoundaryMapping< NB, QL, params.dimension >& map_result,
                                 const Eigen::Matrix< val_t, int{NB}, params.n_fields >&          node_vals,
                                 std::span< const val_t, QL >                                     quad_weights,
                                 val_t time) -> KernelInterface< params >::Rhs
{
    const auto& [J, vals, ders, points, normals] = map_result;
    const auto fields                            = map::FieldValuesAtPoints{*vals, ders, node_vals};
    auto       retval                            = lstr::detail::initResidualKernelResult< params >();
    for (size_t p = 0; p != QL; ++p)
    {
        const auto point              = SpaceTimePoint{points[p], time};
        const auto [field_v, field_d] = fields.get(p);
        const auto wgt                = quad_weights[p] * J[p];
        retval += wgt * kernel({field_v, field_d, point, normals[p]});
    }
    return retval;
}

template < typename Kernel, KernelParams params, el_o_t... orders, AssemblyOptions options >
auto evalLocalIntegral(const ResidualDomainKernel< Kernel, params >& kernel,
                       const mesh::MeshPartition< orders... >&       mesh,
                       const util::ArrayOwner< d_id_t >&             domain_ids,
                       const FieldAccess< params.n_fields >&         field_access,
                       util::ConstexprValue< options >,
                       val_t time) -> KernelInterface< params >::Rhs
{
    const auto reduce_element = [&]< mesh::ElementType ET, el_o_t EO >(
                                    const mesh::Element< ET, EO >& element) -> KernelInterface< params >::Rhs {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            constexpr auto BT = options.basis_type;
            constexpr auto QT = options.quad_type;
            constexpr auto GO = mesh::ElementTraits< mesh::Element< ET, EO > >::geom_order;
            constexpr auto QO = options.order(EO) + GO;

            const auto node_vals    = field_access.getGloballyIndexed(element.nodes);
            const auto basis_at_qps = basis::getQuadratureView< BT, ET, EO, QT, QO >();
            const auto verts        = std::span{element.data.vertices};
            const auto mapping      = map::TabulatedDomainMapping{basis_at_qps.bases, verts};
            return evalElementIntegral(kernel, mapping, node_vals, basis_at_qps.weights, time);
        }
        else
            return lstr::detail::initResidualKernelResult< params >();
    };
    const auto zero = lstr::detail::initResidualKernelResult< params >();
    return mesh.transformReduce(domain_ids, zero, reduce_element, std::plus{}, std::execution::par);
}

template < typename Kernel, KernelParams params, el_o_t... orders, AssemblyOptions options >
auto evalLocalIntegral(const ResidualBoundaryKernel< Kernel, params >& kernel,
                       const mesh::MeshPartition< orders... >&         mesh,
                       const util::ArrayOwner< d_id_t >&               boundary_ids,
                       const FieldAccess< params.n_fields >&           field_access,
                       util::ConstexprValue< options >,
                       val_t time) -> KernelInterface< params >::Rhs
{
    const auto reduce_element =
        [&]< mesh::ElementType ET, el_o_t EO >(
            const mesh::BoundaryElementView< ET, EO >& el_view) -> KernelInterface< params >::Rhs {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            constexpr auto BT = options.basis_type;
            constexpr auto QT = options.quad_type;
            constexpr auto GT = util::ConstexprValue< mesh::ElementTraits< mesh::Element< ET, EO > >::geom_type >{};
            constexpr auto GO = mesh::ElementTraits< mesh::Element< ET, EO > >::geom_order;
            constexpr auto QO = options.order(EO) + GO;

            const auto side         = el_view.getSide();
            const auto node_vals    = field_access.getGloballyIndexed(el_view->nodes);
            const auto basis_at_qps = basis::getSideQuadratureView< BT, ET, EO, QT, QO >(side);
            const auto verts        = std::span{el_view->data.vertices};
            const auto mapping      = map::TabulatedBoundaryMapping{basis_at_qps.bases, verts, GT, side};
            return evalElementBoundaryIntegral(kernel, mapping, node_vals, basis_at_qps.weights, time);
        }
        else
            return lstr::detail::initResidualKernelResult< params >();
    };
    const auto zero = lstr::detail::initResidualKernelResult< params >();
    return mesh.transformReduceBoundaries(boundary_ids, zero, reduce_element, std::plus{}, std::execution::par);
}
} // namespace post

template < ResidualKernel_c Kernel, el_o_t... orders, AssemblyOptions opts = {} >
auto computeIntegral(const MpiComm&                                          comm,
                     const Kernel&                                           kernel,
                     const mesh::MeshPartition< orders... >&                 mesh,
                     const util::ArrayOwner< d_id_t >&                       domain_ids,
                     const post::FieldAccess< Kernel::parameters.n_fields >& field_access = {},
                     util::ConstexprValue< opts >                            opts_ctwrpr  = {},
                     val_t                                                   time         = 0.)
{
    const auto local_integral  = post::evalLocalIntegral(kernel, mesh, domain_ids, field_access, opts_ctwrpr, time);
    auto       global_integral = detail::initResidualKernelResult< Kernel::parameters >();
    auto       comm_view       = std::views::counted(local_integral.data(), Kernel::parameters.n_equations);
    comm.allReduce(std::move(comm_view), global_integral.data(), MPI_SUM);
    return global_integral;
}
} // namespace lstr
#endif // L3STER_POST_INTEGRAL_HPP
