#ifndef L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP
#define L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP

#include "l3ster/assembly/ScatterLocalSystem.hpp"
#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"

#include <iostream>

namespace lstr
{
struct AssemblyOptions
{
    q_o_t           value_order      = 1;
    q_o_t           derivative_order = 0;
    BasisTypes      basis_type       = BasisTypes::Lagrange;
    QuadratureTypes quad_type        = QuadratureTypes::GLeg;
};
} // namespace lstr

#define L3STER_MAKE_ASM_OPTS(...)                                                                                      \
    lstr::ConstexprValue< lstr::AssemblyOptions{__VA_ARGS__} >                                                         \
    {}

namespace lstr
{
namespace detail
{
template < typename F >
struct FieldValGetterDeductionHelper
{
    template < ElementTypes ET, el_o_t EO >
    struct DeductionHelper
    {
        static constexpr bool value =
            std::invocable< F, const typename Element< ET, EO >::node_array_t > and
            EigenMatrix_c< std::invoke_result_t< F, const typename Element< ET, EO >::node_array_t > > and
            (std::invoke_result_t< F, const typename Element< ET, EO >::node_array_t >::RowsAtCompileTime ==
             Element< ET, EO >::n_nodes);
    };
    static constexpr bool value = assert_all_elements< DeductionHelper >;
};
template < typename F >
concept FieldValGetter_c = FieldValGetterDeductionHelper< F >::value;
template < FieldValGetter_c F >
inline constexpr auto deduce_n_fields = std::invoke(
    []< typename... TypeOrderPair >(TypePack< TypeOrderPair... >) {
        constexpr auto deduce_n_fields = []< ElementTypes ET, el_o_t EO >(ValuePack< ET, EO >) {
            return std::invoke_result_t< F, const typename Element< ET, EO >::node_array_t >::ColsAtCompileTime;
        };
        constexpr auto n_fields_for_els =
            std::array< int, sizeof...(TypeOrderPair) >{deduce_n_fields(TypeOrderPair{})...};
        static_assert(std::ranges::adjacent_find(n_fields_for_els, std::not_equal_to<>{}) == end(n_fields_for_els),
                      "The field value getter must get the same number of fields for all element types");
        return n_fields_for_els[0];
    },
    type_order_combinations{});

// Checking whether the kernel is being invoked in a domain where it is valid is fundamentally a run-time endeavor.
// What we can do at compile time is to assert that there at least exists a domain dimension for which it is valid.
template < typename Kernel, size_t n_fields >
struct PotentiallyValidKernelDeductionHelper
{
    template < ElementTypes T, el_o_t O >
    struct DeductionHelperDomain
    {
        static constexpr bool value = Kernel_c< Kernel, Element< T, O >::native_dim, n_fields >;
    };
    static constexpr bool domain = assert_any_element< DeductionHelperDomain >;

    template < ElementTypes T, el_o_t O >
    struct DeductionHelperBoundary
    {
        static constexpr bool value = BoundaryKernel_c< Kernel, Element< T, O >::native_dim, n_fields >;
    };
    static constexpr bool boundary = assert_any_element< DeductionHelperBoundary >;
};
template < typename Kernel, size_t n_fields >
concept PotentiallyValidKernel_c = PotentiallyValidKernelDeductionHelper< Kernel, n_fields >::domain;
template < typename Kernel, size_t n_fields >
concept PotentiallyValidBoundaryKernel_c = PotentiallyValidKernelDeductionHelper< Kernel, n_fields >::boundary;

// During local assembly, when performing rank updates of statically-sized matrices, Eigen allocates data on the stack.
// For larger matrices, this results in stack overflow. The following code estimates the required stack size to be the
// size of the rank update matrix, which has been shown to be sufficient during testing.
template < typename Kernel, size_t n_fields >
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
    static constexpr size_t domain =
        std::invoke([]< typename... DH >(TypePack< DH... >) { return std::ranges::max(std::array{DH::value...}); },
                    parametrize_type_over_element_types_and_orders_t< TypePack, DeductionHelperDomain >{});

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
    static constexpr size_t boundary =
        std::invoke([]< typename... DH >(TypePack< DH... >) { return std::ranges::max(std::array{DH::value...}); },
                    parametrize_type_over_element_types_and_orders_t< TypePack, DeductionHelperBoundary >{});
};

template < typename Kernel, typename FvalGetter >
consteval auto deduceRequiredStackSizeDomain() -> size_t
{
    return detail::KernelStackSizeDeductionHelper< Kernel, detail::deduce_n_fields< FvalGetter > >::domain;
}

template < typename Kernel, typename FvalGetter >
consteval auto deduceRequiredStackSizeBoundary() -> size_t
{
    return detail::KernelStackSizeDeductionHelper< Kernel, detail::deduce_n_fields< FvalGetter > >::boundary;
}
} // namespace detail

inline constexpr auto empty_field_val_getter =
    []< size_t N >(const std::array< n_id_t, N >&) -> EigenRowMajorMatrix< val_t, static_cast< int >(N), 0 >
{
    return {};
};

template < ArrayOf_c< size_t > auto field_inds, size_t dofs_per_node, AssemblyOptions asm_opts = AssemblyOptions{} >
void assembleGlobalSystem(auto&&                                       kernel,
                          const MeshPartition&                         mesh,
                          detail::DomainIdRange_c auto&&               domain_ids,
                          detail::FieldValGetter_c auto&&              fval_getter,
                          tpetra_crsmatrix_t&                          global_matrix,
                          std::span< val_t >                           global_vector,
                          const NodeToLocalDofMap< dofs_per_node, 3 >& dof_map,
                          ConstexprValue< field_inds >                 field_inds_ctwrpr,
                          ConstexprValue< asm_opts >                   assembly_options = {},
                          val_t                                        time             = 0.)
    requires detail::PotentiallyValidKernel_c< decltype(kernel), detail::deduce_n_fields< decltype(fval_getter) > >
{
    L3STER_PROFILE_FUNCTION;
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        constexpr auto el_dim   = Element< ET, EO >::native_dim;
        constexpr auto n_fields = detail::deduce_n_fields< decltype(fval_getter) >;
        if constexpr (detail::Kernel_c< decltype(kernel), el_dim, n_fields >)
        {
            constexpr auto  BT             = asm_opts.basis_type;
            constexpr auto  QT             = asm_opts.quad_type;
            constexpr q_o_t QO             = 2 * (asm_opts.value_order * EO + asm_opts.derivative_order * (EO - 1));
            const auto      field_vals     = fval_getter(element.getNodes());
            const auto&     qbv            = getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
            const auto& [loc_mat, loc_vec] = assembleLocalSystem(kernel, element, field_vals, qbv, time);
            const auto [row_dofs, col_dofs, rhs_dofs] = detail::getUnsortedElementDofs< field_inds >(element, dof_map);
            detail::scatterLocalSystem(loc_mat, loc_vec, global_matrix, global_vector, row_dofs, col_dofs, rhs_dofs);
        }
        else
        {
            std::cerr << "Attempting to assemble local system for which the passed kernel is invalid. Please check the "
                         "kernel was defined correctly, and that you are assembling the problem in the correct domain "
                         "(e.g. that you're not trying to assemble a 2D problem in a 3D domain). This process will now "
                         "terminate.\n";
            std::terminate(); // Throwing in a parallel context would terminate regardless
        }
    };
    const auto stack_size_guard = util::StackSizeGuard{
        util::default_stack_size + detail::deduceRequiredStackSizeDomain< decltype(kernel), decltype(fval_getter) >()};
    mesh.visit(process_element, std::forward< decltype(domain_ids) >(domain_ids), std::execution::par);
}

template < ArrayOf_c< size_t > auto field_inds, size_t dofs_per_node, AssemblyOptions asm_opts = AssemblyOptions{} >
void assembleGlobalBoundarySystem(auto&&                                       kernel,
                                  const BoundaryView&                          boundary,
                                  detail::FieldValGetter_c auto&&              fval_getter,
                                  tpetra_crsmatrix_t&                          global_matrix,
                                  std::span< val_t >                           global_vector,
                                  const NodeToLocalDofMap< dofs_per_node, 3 >& dof_map,
                                  ConstexprValue< field_inds >                 field_inds_ctwrpr,
                                  ConstexprValue< asm_opts >                   assembly_options = {},
                                  val_t                                        time             = 0.)
    requires detail::PotentiallyValidBoundaryKernel_c< decltype(kernel),
                                                       detail::deduce_n_fields< decltype(fval_getter) > >
{
    L3STER_PROFILE_FUNCTION;
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) {
        constexpr auto el_dim   = Element< ET, EO >::native_dim;
        constexpr auto n_fields = detail::deduce_n_fields< decltype(fval_getter) >;
        if constexpr (detail::BoundaryKernel_c< decltype(kernel), el_dim, n_fields >)
        {
            constexpr auto  BT         = asm_opts.basis_type;
            constexpr auto  QT         = asm_opts.quad_type;
            constexpr q_o_t QO         = 2 * (asm_opts.value_order * EO + asm_opts.derivative_order * (EO - 1));
            const auto      field_vals = fval_getter(el_view->getNodes());
            const auto&     qbv        = getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(el_view.getSide());
            const auto& [loc_mat, loc_vec] = assembleLocalBoundarySystem(kernel, el_view, field_vals, qbv, time);
            const auto [row_dofs, col_dofs, rhs_dofs] = detail::getUnsortedElementDofs< field_inds >(*el_view, dof_map);
            detail::scatterLocalSystem(loc_mat, loc_vec, global_matrix, global_vector, row_dofs, col_dofs, rhs_dofs);
        }
        else
        {
            std::cerr << "Attempting to assemble local boundary system for which the passed kernel is invalid. Please "
                         "check the kernel was defined correctly, and that you are assembling the problem in the "
                         "correct domain (e.g. that you're not trying to assemble a 2D problem in a 3D domain). This "
                         "process will now terminate.\n";
            std::terminate(); // Throwing in a parallel context would terminate regardless
        };
    };
    const auto stack_size_guard =
        util::StackSizeGuard{util::default_stack_size +
                             detail::deduceRequiredStackSizeBoundary< decltype(kernel), decltype(fval_getter) >()};
    boundary.visit(process_element, std::execution::par);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP
