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
        requires std::invocable< F, const typename Element< ET, EO >::node_array_t > and
                 EigenMatrix_c< std::invoke_result_t< F, const typename Element< ET, EO >::node_array_t > > and
                 (std::invoke_result_t< F, const typename Element< ET, EO >::node_array_t >::RowsAtCompileTime ==
                  Element< ET, EO >::n_nodes)
    struct DeductionHelper : std::true_type
    {};
    static constexpr bool value = assert_all_elements< DeductionHelper >;
};
template < typename F >
concept FieldValGetter_c = FieldValGetterDeductionHelper< F >::value;
template < FieldValGetter_c F >
inline constexpr auto deduce_n_fields = []< typename... TypeOrderPair >(TypePack< TypeOrderPair... >) {
    constexpr auto deduce_n_fields = []< ElementTypes ET, el_o_t EO >(ValuePack< ET, EO >) {
        return std::invoke_result_t< F, const typename Element< ET, EO >::node_array_t >::ColsAtCompileTime;
    };
    constexpr auto n_fields_for_els = std::array< int, sizeof...(TypeOrderPair) >{deduce_n_fields(TypeOrderPair{})...};
    static_assert(std::ranges::adjacent_find(n_fields_for_els, std::not_equal_to<>{}) == end(n_fields_for_els),
                  "The field value getter must get the same number of fields for all element types");
    return n_fields_for_els[0];
}(type_order_combinations{});
} // namespace detail

template < BasisTypes              BT,
           QuadratureTypes         QT,
           q_o_t                   QO,
           array_of< size_t > auto field_inds,
           typename Kernel,
           detail::FieldValGetter_c FvalGetter,
           std::ranges::sized_range R,
           size_t                   n_fields >
void assembleGlobalSystem(Kernel&&                                               kernel,
                          const MeshPartition&                                   mesh,
                          R&&                                                    domain_ids,
                          const NodeToDofMap< n_fields >&                        map,
                          FvalGetter&&                                           field_val_getter,
                          Tpetra::CrsMatrix< val_t, local_dof_t, global_dof_t >& global_matrix,
                          const Teuchos::ArrayRCP< val_t >&                      global_vector,
                          const Tpetra::Map< local_dof_t, global_dof_t >&        dof_local_global_map,
                          val_t                                                  time = 0.)
{
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        constexpr auto el_dim = Element< ET, EO >::native_dim;
        if constexpr (detail::Kernel_c< Kernel, el_dim, detail::deduce_n_fields< FvalGetter > >)
        {
            const auto  element_dofs                = detail::getUnsortedElementDofs< field_inds >(element, map);
            const auto  field_vals                  = field_val_getter(element.getNodes());
            const auto& qbv                         = getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
            const auto [local_matrix, local_vector] = assembleLocalSystem(kernel, element, field_vals, qbv, time);
            detail::scatterLocalMatrix(local_matrix, element_dofs, global_matrix);
            detail::scatterLocalVector(local_vector, element_dofs, global_vector, dof_local_global_map);
        }
    };
    mesh.visit(process_element, std::forward< R >(domain_ids), std::execution::par);
}

template < BasisTypes              BT,
           QuadratureTypes         QT,
           q_o_t                   QO,
           array_of< size_t > auto field_inds,
           typename Kernel,
           detail::FieldValGetter_c FvalGetter,
           size_t                   n_fields >
void assembleGlobalBoundarySystem(Kernel&&                                               kernel,
                                  const BoundaryView&                                    boundary,
                                  const NodeToDofMap< n_fields >&                        map,
                                  FvalGetter&&                                           field_val_getter,
                                  Tpetra::CrsMatrix< val_t, local_dof_t, global_dof_t >& global_matrix,
                                  const Teuchos::ArrayRCP< val_t >&                      global_vector,
                                  const Tpetra::Map< local_dof_t, global_dof_t >&        dof_local_global_map,
                                  val_t                                                  time = 0.)
{
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) {
        constexpr auto el_dim = Element< ET, EO >::native_dim;
        if constexpr (detail::BoundaryKernel_c< Kernel, el_dim, detail::deduce_n_fields< FvalGetter > >)
        {
            const auto  element_dofs = detail::getUnsortedElementDofs< field_inds >(*el_view.element, map);
            const auto  field_vals   = field_val_getter(el_view.element->getNodes());
            const auto& qbv = getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(el_view.element_side);
            const auto [local_matrix, local_vector] =
                assembleLocalBoundarySystem(kernel, el_view, field_vals, qbv, time);
            detail::scatterLocalMatrix(local_matrix, element_dofs, global_matrix);
            detail::scatterLocalVector(local_vector, element_dofs, global_vector, dof_local_global_map);
        }
    };
    boundary.visit(process_element, std::execution::par);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ASSEMBLEGLOBALSYSTEM_HPP