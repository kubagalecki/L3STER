#ifndef L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP
#define L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP

#include "l3ster/assembly/ScatterLocalSystem.hpp"
#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"

namespace lstr
{
namespace detail
{
template < typename F >
struct FieldValGetterDeductionHelper
{
    template < ElementTypes ET, el_o_t EO >
    struct DeductionHelper : std::false_type
    {};
    template < ElementTypes ET, el_o_t EO >
        requires std::invocable< F, const typename Element< ET, EO >::node_array_t > and
                 EigenMatrix_c< std::invoke_result_t< F, const typename Element< ET, EO >::node_array_t > > and
                 (std::invoke_result_t< F, const typename Element< ET, EO >::node_array_t >::RowsAtCompileTime ==
                  Element< ET, EO >::n_nodes)
    struct DeductionHelper< ET, EO > : std::true_type
    {};
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
} // namespace detail

inline constexpr detail::EmptyFieldValGetter empty_field_val_getter{};

template < BasisTypes               BT,
           QuadratureTypes          QT,
           q_o_t                    QO,
           ArrayOf_c< size_t > auto field_inds,
           typename Kernel,
           detail::DomainIdRange_c R,
           size_t                  n_fields >
void assembleGlobalSystem(Kernel&&                                               kernel,
                          const MeshPartition&                                   mesh,
                          R&&                                                    domain_ids,
                          detail::FieldValGetter_c auto&&                        field_val_getter,
                          Tpetra::CrsMatrix< val_t, local_dof_t, global_dof_t >& global_matrix,
                          std::span< val_t >                                     global_vector,
                          const NodeToLocalDofMap< n_fields >&                   row_map,
                          const NodeToLocalDofMap< n_fields >&                   col_map,
                          const NodeToLocalDofMap< n_fields >&                   rhs_map,
                          val_t                                                  time = 0.)
{
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        constexpr auto el_dim = Element< ET, EO >::native_dim;
        if constexpr (detail::Kernel_c< Kernel, el_dim, detail::deduce_n_fields< decltype(field_val_getter) > >)
        {
            const auto  field_vals        = field_val_getter(element.getNodes());
            const auto& qbv               = getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
            const auto [loc_mat, loc_vec] = assembleLocalSystem(kernel, element, field_vals, qbv, time);
            const auto row_dofs           = detail::getUnsortedElementDofs< field_inds >(element, row_map);
            const auto col_dofs           = detail::getUnsortedElementDofs< field_inds >(element, col_map);
            const auto rhs_dofs           = detail::getUnsortedElementDofs< field_inds >(element, rhs_map);
            detail::scatterLocalSystem(loc_mat, loc_vec, global_matrix, global_vector, row_dofs, col_dofs, rhs_dofs);
        }
    };
    mesh.visit(process_element, std::forward< R >(domain_ids), std::execution::par);
}

template < BasisTypes               BT,
           QuadratureTypes          QT,
           q_o_t                    QO,
           ArrayOf_c< size_t > auto field_inds,
           typename Kernel,
           size_t n_fields >
void assembleGlobalBoundarySystem(Kernel&&                                               kernel,
                                  const BoundaryView&                                    boundary,
                                  detail::FieldValGetter_c auto&&                        field_val_getter,
                                  Tpetra::CrsMatrix< val_t, local_dof_t, global_dof_t >& global_matrix,
                                  std::span< val_t >                                     global_vector,
                                  const NodeToLocalDofMap< n_fields >&                   row_map,
                                  const NodeToLocalDofMap< n_fields >&                   col_map,
                                  const NodeToLocalDofMap< n_fields >&                   rhs_map,
                                  val_t                                                  time = 0.)
{
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) {
        constexpr auto el_dim = Element< ET, EO >::native_dim;
        if constexpr (detail::BoundaryKernel_c< Kernel, el_dim, detail::deduce_n_fields< decltype(field_val_getter) > >)
        {
            const auto  field_vals = field_val_getter(el_view->getNodes());
            const auto& qbv        = getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(el_view.getSide());
            const auto [loc_mat, loc_vec] = assembleLocalBoundarySystem(kernel, el_view, field_vals, qbv, time);
            const auto row_dofs           = detail::getUnsortedElementDofs< field_inds >(*el_view, row_map);
            const auto col_dofs           = detail::getUnsortedElementDofs< field_inds >(*el_view, col_map);
            const auto rhs_dofs           = detail::getUnsortedElementDofs< field_inds >(*el_view, rhs_map);
            detail::scatterLocalSystem(loc_mat, loc_vec, global_matrix, global_vector, row_dofs, col_dofs, rhs_dofs);
        }
    };
    boundary.visit(process_element, std::execution::par);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP
