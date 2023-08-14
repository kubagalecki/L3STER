#ifndef L3STER_GLOB_ASM_ASSEMBLEGLOBALSYSTEM_HPP
#define L3STER_GLOB_ASM_ASSEMBLEGLOBALSYSTEM_HPP

#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"
#include "l3ster/glob_asm/StaticCondensationManager.hpp"
#include "l3ster/post/SolutionManager.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include <iostream>

namespace lstr
{
struct AssemblyOptions
{
    q_o_t                value_order      = 1;
    q_o_t                derivative_order = 0;
    basis::BasisType     basis_type       = basis::BasisType::Lagrange;
    quad::QuadratureType quad_type        = quad::QuadratureType::GaussLegendre;

    [[nodiscard]] constexpr q_o_t order(el_o_t elem_order) const
    {
        return static_cast< q_o_t >(value_order * elem_order + derivative_order * (elem_order - 1));
    }
};
} // namespace lstr

namespace lstr::glob_asm
{
template < typename Kernel,
           KernelParams params,
           el_o_t... orders,
           ArrayOf_c< size_t > auto field_inds,
           size_t                   dofs_per_node,
           CondensationPolicy       CP,
           AssemblyOptions          asm_opts >
void assembleGlobalSystem(const DomainKernel< Kernel, params >&                       kernel,
                          const mesh::MeshPartition< orders... >&                     mesh,
                          mesh::DomainIdRange_c auto&&                                domain_ids,
                          const SolutionManager::FieldValueGetter< params.n_fields >& fval_getter,
                          tpetra_crsmatrix_t&                                         global_mat,
                          std::span< val_t >                                          global_rhs,
                          const dofs::NodeToLocalDofMap< dofs_per_node, 3 >&          dof_map,
                          StaticCondensationManager< CP >&                            condensation_manager,
                          util::ConstexprValue< field_inds >                          field_inds_ctwrpr,
                          util::ConstexprValue< asm_opts >,
                          val_t time = 0.)
{
    L3STER_PROFILE_FUNCTION;

    const bool dim_match = std::ranges::all_of(domain_ids, [&](auto id) {
        try
        {
            const auto dom_view = mesh.getDomainView(id);
            return dom_view.getDim() == params.dimension;
        }
        catch (const std::out_of_range&) // Domain not present in partition means kernel will not be invoked
        {
            return true;
        }
    });
    util::throwingAssert(dim_match, "The dimension of the kernel does not match the dimension of the domain");

    const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            constexpr auto  BT             = asm_opts.basis_type;
            constexpr auto  QT             = asm_opts.quad_type;
            constexpr q_o_t QO             = 2 * asm_opts.order(EO);
            const auto      field_vals     = fval_getter(element.getNodes());
            const auto&     rbq            = basis::getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
            const auto& [loc_mat, loc_rhs] = assembleLocalSystem(kernel, element, field_vals, rbq, time);
            condensation_manager.condenseSystem(
                dof_map, global_mat, global_rhs, loc_mat, loc_rhs, element, field_inds_ctwrpr);
        }
    };
    const auto max_par_guard = util::MaxParallelismGuard{};
    mesh.visit(process_element, std::forward< decltype(domain_ids) >(domain_ids), std::execution::par);
}

template < typename Kernel,
           KernelParams params,
           el_o_t... orders,
           ArrayOf_c< size_t > auto field_inds,
           size_t                   dofs_per_node,
           CondensationPolicy       CP,
           AssemblyOptions          asm_opts >
void assembleGlobalBoundarySystem(const BoundaryKernel< Kernel, params >&                     kernel,
                                  const mesh::BoundaryView< orders... >&                      boundary,
                                  const SolutionManager::FieldValueGetter< params.n_fields >& fval_getter,
                                  tpetra_crsmatrix_t&                                         global_mat,
                                  std::span< val_t >                                          global_rhs,
                                  const dofs::NodeToLocalDofMap< dofs_per_node, 3 >&          dof_map,
                                  StaticCondensationManager< CP >&                            condensation_manager,
                                  util::ConstexprValue< field_inds >                          field_inds_ctwrpr,
                                  util::ConstexprValue< asm_opts >,
                                  val_t time = 0.)
{
    L3STER_PROFILE_FUNCTION;

    const bool dim_match = boundary.reduce(
        true,
        [&]< mesh::ElementType ET, el_o_t EO >(mesh::BoundaryElementView< ET, EO >) {
            return mesh::Element< ET, EO >::native_dim == params.dimension;
        },
        std::logical_and{});
    util::throwingAssert(dim_match, "The dimension of the kernel does not match the dimension of the domain");

    const auto process_element =
        [&]< mesh::ElementType ET, el_o_t EO >(const mesh::BoundaryElementView< ET, EO >& el_view) {
            if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
            {
                constexpr auto  BT         = asm_opts.basis_type;
                constexpr auto  QT         = asm_opts.quad_type;
                constexpr q_o_t QO         = 2 * asm_opts.order(EO);
                const auto      field_vals = fval_getter(el_view->getNodes());
                const auto& qbv = basis::getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(el_view.getSide());
                const auto& [loc_mat, loc_rhs] = assembleLocalBoundarySystem(kernel, el_view, field_vals, qbv, time);
                condensation_manager.condenseSystem(
                    dof_map, global_mat, global_rhs, loc_mat, loc_rhs, *el_view, field_inds_ctwrpr);
            }
        };
    const auto max_par_guard = util::MaxParallelismGuard{};
    boundary.visit(process_element, std::execution::par);
}
} // namespace lstr::glob_asm
#endif // L3STER_GLOB_ASM_ASSEMBLEGLOBALSYSTEM_HPP
