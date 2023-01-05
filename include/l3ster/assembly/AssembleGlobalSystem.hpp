#ifndef L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP
#define L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP

#include "l3ster/assembly/ScatterLocalSystem.hpp"
#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"

#include <iostream>

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

struct EmptyFieldValGetter
{
    template < size_t N >
    auto operator()(const std::array< n_id_t, N >&) const -> EigenRowMajorMatrix< val_t, N, 0 >
    {
        return {};
    }
};

// Checking whether the kernel is being invoked in a domain where it is valid is fundamentally a run-time endeavour.
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
} // namespace detail

inline constexpr detail::EmptyFieldValGetter empty_field_val_getter{};

template < BasisTypes BT, QuadratureTypes QT, q_o_t QO, ArrayOf_c< size_t > auto field_inds, size_t dofs_per_node >
void assembleGlobalSystem(auto&&                                       kernel,
                          const MeshPartition&                         mesh,
                          detail::DomainIdRange_c auto&&               domain_ids,
                          detail::FieldValGetter_c auto&&              field_val_getter,
                          tpetra_crsmatrix_t&                          global_matrix,
                          std::span< val_t >                           global_vector,
                          const NodeToLocalDofMap< dofs_per_node, 3 >& dof_map,
                          val_t                                        time = 0.)
    requires detail::PotentiallyValidKernel_c< decltype(kernel), detail::deduce_n_fields< decltype(field_val_getter) > >
{
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        constexpr auto el_dim   = Element< ET, EO >::native_dim;
        constexpr auto n_fields = detail::deduce_n_fields< decltype(field_val_getter) >;
        if constexpr (detail::Kernel_c< decltype(kernel), el_dim, n_fields >)
        {
            const auto  field_vals                    = field_val_getter(element.getNodes());
            const auto& qbv                           = getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
            const auto  local_system                  = assembleLocalSystem(kernel, element, field_vals, qbv, time);
            const auto& [loc_mat, loc_vec]            = *local_system;
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
    mesh.visit(process_element, std::forward< decltype(domain_ids) >(domain_ids), std::execution::par);
}

template < BasisTypes BT, QuadratureTypes QT, q_o_t QO, ArrayOf_c< size_t > auto field_inds, size_t dofs_per_node >
void assembleGlobalBoundarySystem(auto&&                                       kernel,
                                  const BoundaryView&                          boundary,
                                  detail::FieldValGetter_c auto&&              field_val_getter,
                                  tpetra_crsmatrix_t&                          global_matrix,
                                  std::span< val_t >                           global_vector,
                                  const NodeToLocalDofMap< dofs_per_node, 3 >& dof_map,
                                  val_t                                        time = 0.)
    requires detail::PotentiallyValidBoundaryKernel_c< decltype(kernel),
                                                       detail::deduce_n_fields< decltype(field_val_getter) > >
{
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) {
        constexpr auto el_dim   = Element< ET, EO >::native_dim;
        constexpr auto n_fields = detail::deduce_n_fields< decltype(field_val_getter) >;
        if constexpr (detail::BoundaryKernel_c< decltype(kernel), el_dim, n_fields >)
        {
            const auto  field_vals   = field_val_getter(el_view->getNodes());
            const auto& qbv          = getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(el_view.getSide());
            const auto  local_system = assembleLocalBoundarySystem(kernel, el_view, field_vals, qbv, time);
            const auto& [loc_mat, loc_vec]            = *local_system;
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
    boundary.visit(process_element, std::execution::par);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP
