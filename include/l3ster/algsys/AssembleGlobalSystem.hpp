#ifndef L3STER_ALGSYS_ASSEMBLEGLOBALSYSTEM_HPP
#define L3STER_ALGSYS_ASSEMBLEGLOBALSYSTEM_HPP

#include "l3ster/algsys/OperatorUtils.hpp"
#include "l3ster/algsys/StaticCondensationManager.hpp"
#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"
#include "l3ster/mesh/BoundaryView.hpp"
#include "l3ster/post/FieldAccess.hpp"
#include "l3ster/util/ScopeGuards.hpp"

namespace lstr::algsys
{
template < typename Kernel,
           KernelParams params,
           el_o_t... orders,
           ArrayOf_c< size_t > auto field_inds,
           size_t                   dofs_per_node,
           CondensationPolicy       CP,
           AssemblyOptions          asm_opts >
void assembleGlobalSystem(const DomainEquationKernel< Kernel, params >&         kernel,
                          const mesh::MeshPartition< orders... >&               mesh,
                          const util::ArrayOwner< d_id_t >&                     domain_ids,
                          const post::FieldAccess< params.n_fields >&           field_access,
                          tpetra_crsmatrix_t&                                   global_mat,
                          const std::array< std::span< val_t >, params.n_rhs >& global_rhs,
                          const dofs::NodeToLocalDofMap< dofs_per_node, 3 >&    dof_map,
                          StaticCondensationManager< CP >&                      condensation_manager,
                          util::ConstexprValue< field_inds >                    field_inds_ctwrpr,
                          util::ConstexprValue< asm_opts >,
                          val_t time = 0.)
{
    L3STER_PROFILE_FUNCTION;

    const bool dim_match = detail::checkDomainDimension(mesh, domain_ids, params.dimension);
    util::throwingAssert(dim_match, "The dimensions of the kernel do not match the dimensions of the domain");

    const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            constexpr auto  BT             = asm_opts.basis_type;
            constexpr auto  QT             = asm_opts.quad_type;
            constexpr q_o_t QO             = 2 * asm_opts.order(EO);
            const auto      field_vals     = field_access.getGloballyIndexed(element.nodes);
            const auto&     rbq            = basis::getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
            const auto& [loc_mat, loc_rhs] = assembleLocalSystem(kernel, element, field_vals, rbq, time);
            condensation_manager.condenseSystem(
                dof_map, global_mat, global_rhs, loc_mat, loc_rhs, element, field_inds_ctwrpr);
        }
    };
    const auto n_cores       = util::GlobalResource< util::hwloc::Topology >::getMaybeUninitialized().getNCores();
    const auto max_par_guard = util::MaxParallelismGuard{n_cores};
    mesh.visit(process_element, domain_ids, std::execution::par);
}

template < typename Kernel,
           KernelParams params,
           el_o_t... orders,
           ArrayOf_c< size_t > auto field_inds,
           size_t                   dofs_per_node,
           CondensationPolicy       CP,
           AssemblyOptions          asm_opts >
void assembleGlobalSystem(const BoundaryEquationKernel< Kernel, params >&       kernel,
                          const mesh::MeshPartition< orders... >&               mesh,
                          const util::ArrayOwner< d_id_t >&                     boundary_ids,
                          const post::FieldAccess< params.n_fields >&           field_access,
                          tpetra_crsmatrix_t&                                   global_mat,
                          const std::array< std::span< val_t >, params.n_rhs >& global_rhs,
                          const dofs::NodeToLocalDofMap< dofs_per_node, 3 >&    dof_map,
                          StaticCondensationManager< CP >&                      condensation_manager,
                          util::ConstexprValue< field_inds >                    field_inds_ctwrpr,
                          util::ConstexprValue< asm_opts >,
                          val_t time = 0.)
{
    L3STER_PROFILE_FUNCTION;

    const bool dim_match = detail::checkDomainDimension(mesh, boundary_ids, params.dimension - 1);
    util::throwingAssert(dim_match, "The dimensions of the kernel do not match the dimensions of the boundary");

    const auto process_element =
        [&]< mesh::ElementType ET, el_o_t EO >(const mesh::BoundaryElementView< ET, EO >& el_view) {
            if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
            {
                constexpr auto  BT         = asm_opts.basis_type;
                constexpr auto  QT         = asm_opts.quad_type;
                constexpr q_o_t QO         = 2 * asm_opts.order(EO);
                const auto      field_vals = field_access.getGloballyIndexed(el_view->nodes);
                const auto& qbv = basis::getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(el_view.getSide());
                const auto& [loc_mat, loc_rhs] = assembleLocalSystem(kernel, el_view, field_vals, qbv, time);
                condensation_manager.condenseSystem(
                    dof_map, global_mat, global_rhs, loc_mat, loc_rhs, *el_view, field_inds_ctwrpr);
            }
        };
    const auto n_cores       = util::GlobalResource< util::hwloc::Topology >::getMaybeUninitialized().getNCores();
    const auto max_par_guard = util::MaxParallelismGuard{n_cores};
    mesh.visitBoundaries(process_element, boundary_ids, std::execution::par);
}
} // namespace lstr::algsys
#endif // L3STER_ALGSYS_ASSEMBLEGLOBALSYSTEM_HPP
