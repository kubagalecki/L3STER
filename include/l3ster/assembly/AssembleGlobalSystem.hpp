#ifndef L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP
#define L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP

#include "l3ster/assembly/SolutionManager.hpp"
#include "l3ster/assembly/StaticCondensationManager.hpp"
#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"
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

#define L3STER_MAKE_ASM_OPTS(...)                                                                                      \
    lstr::ConstexprValue< lstr::AssemblyOptions{__VA_ARGS__} >                                                         \
    {}

namespace lstr
{
namespace detail
{
// Checking whether the kernel is being invoked in a domain where it is valid is fundamentally a run-time endeavor.
// What we can do at compile time is to assert that there at least exists a domain dimension for which it is valid.
template < typename Kernel, size_t n_fields, el_o_t... orders >
struct PotentiallyValidKernelDeductionHelper
{
    template < ElementTypes T, el_o_t O >
    struct DeductionHelperDomain
    {
        static constexpr bool value = Kernel_c< Kernel, Element< T, O >::native_dim, n_fields >;
    };
    static constexpr bool domain =
        ElementDeductionHelper< orders... >::template assert_any_element< DeductionHelperDomain >;

    template < ElementTypes T, el_o_t O >
    struct DeductionHelperBoundary
    {
        static constexpr bool value = BoundaryKernel_c< Kernel, Element< T, O >::native_dim, n_fields >;
    };
    static constexpr bool boundary =
        ElementDeductionHelper< orders... >::template assert_any_element< DeductionHelperBoundary >;
};
template < typename Kernel, size_t n_fields, el_o_t... orders >
concept PotentiallyValidKernel_c = PotentiallyValidKernelDeductionHelper< Kernel, n_fields, orders... >::domain;
template < typename Kernel, size_t n_fields, el_o_t... orders >
concept PotentiallyValidBoundaryKernel_c =
    PotentiallyValidKernelDeductionHelper< Kernel, n_fields, orders... >::boundary;

// During local assembly, when performing rank updates of statically-sized matrices, Eigen allocates data on the stack.
// For larger matrices, this results in stack overflow. The following code estimates the required stack size to be the
// size of the rank update matrix, which has been shown to be sufficient during testing.
template < typename Kernel, size_t n_fields, el_o_t... orders >
struct KernelStackSizeDeductionHelper
{
    template < ElementTypes ET, el_o_t EO >
    struct DeductionHelperDomain
    {
        static constexpr size_t value = std::invoke([] {
            if constexpr (Kernel_c< Kernel, Element< ET, EO >::native_dim, n_fields >)
                return std::decay_t<
                    decltype(getLocalSystemManager< Kernel, ET, EO, n_fields >()) >::required_stack_size;
            else
                return 0;
        });
    };
    static constexpr size_t domain = std::invoke(
        []< typename... DH >(util::TypePack< DH... >) { return std::ranges::max(std::array{DH::value...}); },
        parametrize_type_over_element_types_and_orders_t< util::TypePack, DeductionHelperDomain, orders... >{});

    template < ElementTypes ET, el_o_t EO >
    struct DeductionHelperBoundary
    {
        static constexpr size_t value = std::invoke([] {
            if constexpr (BoundaryKernel_c< Kernel, Element< ET, EO >::native_dim, n_fields >)
                return std::decay_t<
                    decltype(getLocalSystemManager< Kernel, ET, EO, n_fields >()) >::required_stack_size;
            else
                return 0;
        });
    };
    static constexpr size_t boundary = std::invoke(
        []< typename... DH >(util::TypePack< DH... >) { return std::ranges::max(std::array{DH::value...}); },
        parametrize_type_over_element_types_and_orders_t< util::TypePack, DeductionHelperBoundary, orders... >{});
};

template < typename Kernel, size_t n_fields, el_o_t... orders >
consteval auto deduceRequiredStackSizeDomain() -> size_t
{
    return detail::KernelStackSizeDeductionHelper< Kernel, n_fields, orders... >::domain;
}

template < typename Kernel, size_t n_fields, el_o_t... orders >
consteval auto deduceRequiredStackSizeBoundary() -> size_t
{
    return detail::KernelStackSizeDeductionHelper< Kernel, n_fields, orders... >::boundary;
}
} // namespace detail

template < el_o_t... orders,
           size_t                   n_fields,
           ArrayOf_c< size_t > auto field_inds,
           size_t                   dofs_per_node,
           CondensationPolicy       CP,
           AssemblyOptions          asm_opts >
void assembleGlobalSystem(auto&&                                               kernel,
                          const MeshPartition< orders... >&                    mesh,
                          detail::DomainIdRange_c auto&&                       domain_ids,
                          const SolutionManager::FieldValueGetter< n_fields >& fval_getter,
                          tpetra_crsmatrix_t&                                  global_mat,
                          std::span< val_t >                                   global_rhs,
                          const NodeToLocalDofMap< dofs_per_node, 3 >&         dof_map,
                          detail::StaticCondensationManager< CP >&             condensation_manager,
                          util::ConstexprValue< field_inds >                   field_inds_ctwrpr,
                          util::ConstexprValue< asm_opts >,
                          val_t time = 0.)
    requires detail::PotentiallyValidKernel_c< decltype(kernel), n_fields, orders... >
{
    L3STER_PROFILE_FUNCTION;
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        constexpr auto el_dim = Element< ET, EO >::native_dim;
        if constexpr (detail::Kernel_c< decltype(kernel), el_dim, n_fields >)
        {
            constexpr auto  BT             = asm_opts.basis_type;
            constexpr auto  QT             = asm_opts.quad_type;
            constexpr q_o_t QO             = 2 * asm_opts.order(EO);
            const auto      field_vals     = fval_getter(element.getNodes());
            const auto&     qbv            = basis::getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
            const auto& [loc_mat, loc_rhs] = assembleLocalSystem(kernel, element, field_vals, qbv, time);
            condensation_manager.condenseSystem(
                dof_map, global_mat, global_rhs, loc_mat, loc_rhs, element, field_inds_ctwrpr);
        }
        else
        {
            // Throwing in a parallel context would terminate regardless
            util::terminatingAssert(
                false,
                "Attempting to assemble local system for which the passed kernel is invalid. Please check that the "
                "kernel was defined correctly, and that you are assembling the problem in the correct domain (e.g. "
                "that you're not trying to assemble a 2D problem in a 3D domain). This process will now terminate.\n");
        }
    };
    constexpr auto required_stack_size =
        util::default_stack_size + detail::deduceRequiredStackSizeDomain< decltype(kernel), n_fields, orders... >();
    util::requestStackSize< required_stack_size >();
    const auto max_par_guard = util::MaxParallelismGuard{};
    mesh.visit(process_element, std::forward< decltype(domain_ids) >(domain_ids), std::execution::par);
}

template < el_o_t... orders,
           size_t                   n_fields,
           ArrayOf_c< size_t > auto field_inds,
           size_t                   dofs_per_node,
           CondensationPolicy       CP,
           AssemblyOptions          asm_opts >
void assembleGlobalBoundarySystem(auto&&                                               kernel,
                                  const BoundaryView< orders... >&                     boundary,
                                  const SolutionManager::FieldValueGetter< n_fields >& fval_getter,
                                  tpetra_crsmatrix_t&                                  global_mat,
                                  std::span< val_t >                                   global_rhs,
                                  const NodeToLocalDofMap< dofs_per_node, 3 >&         dof_map,
                                  detail::StaticCondensationManager< CP >&             condensation_manager,
                                  util::ConstexprValue< field_inds >                   field_inds_ctwrpr,
                                  util::ConstexprValue< asm_opts >,
                                  val_t time = 0.)
    requires detail::PotentiallyValidBoundaryKernel_c< decltype(kernel), n_fields, orders... >
{
    L3STER_PROFILE_FUNCTION;
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) {
        constexpr auto el_dim = Element< ET, EO >::native_dim;
        if constexpr (detail::BoundaryKernel_c< decltype(kernel), el_dim, n_fields >)
        {
            constexpr auto  BT         = asm_opts.basis_type;
            constexpr auto  QT         = asm_opts.quad_type;
            constexpr q_o_t QO         = 2 * asm_opts.order(EO);
            const auto      field_vals = fval_getter(el_view->getNodes());
            const auto&     qbv = basis::getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(el_view.getSide());
            const auto& [loc_mat, loc_rhs] = assembleLocalBoundarySystem(kernel, el_view, field_vals, qbv, time);
            condensation_manager.condenseSystem(
                dof_map, global_mat, global_rhs, loc_mat, loc_rhs, *el_view, field_inds_ctwrpr);
        }
        else
        {
            // Throwing in a parallel context would terminate regardless
            util::terminatingAssert(
                false,
                "Attempting to assemble local boundary system for which the passed kernel is invalid. Please check "
                "that the kernel was defined correctly, and that you are assembling the problem in the correct domain "
                "(e.g. that you're not trying to assemble a 2D problem in a 3D domain). This process will now "
                "terminate.\n");
        };
    };
    constexpr auto required_stack_size =
        util::default_stack_size + detail::deduceRequiredStackSizeBoundary< decltype(kernel), n_fields, orders... >();
    util::requestStackSize< required_stack_size >();
    const auto max_par_guard = util::MaxParallelismGuard{};
    boundary.visit(process_element, std::execution::par);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP
