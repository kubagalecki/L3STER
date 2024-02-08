#ifndef L3STER_MESH_DOMAIN_HPP
#define L3STER_MESH_DOMAIN_HPP

#include "l3ster/common/Enums.hpp"
#include "l3ster/mesh/Element.hpp"
#include "l3ster/mesh/ElementMeta.hpp"
#include "l3ster/util/Common.hpp"
#include "l3ster/util/UniVector.hpp"

namespace lstr::mesh
{
template < typename F, el_o_t... orders >
concept MutableElementVisitor_c = ElementDeductionHelper< orders... >::template invocable_on_elements< F >;
template < typename F, el_o_t... orders >
concept ConstElementVisitor_c = ElementDeductionHelper< orders... >::template invocable_on_const_elements< F >;
template < typename F, el_o_t... orders >
concept ElementPredicate_c =
    ElementDeductionHelper< orders... >::template invocable_on_const_elements_return< bool, F >;
template < typename Zero, typename Transform, typename Reduction, el_o_t... orders >
concept TransformReducible_c =
    ElementDeductionHelper< orders... >::template invocable_on_const_elements_return< Zero, Transform > and
    ReductionFor_c< Reduction, Zero >;

template < el_o_t... orders >
struct Domain
{
    static_assert(sizeof...(orders) > 0);

    static constexpr auto uninitialized_dim = std::numeric_limits< dim_t >::max();

    using el_univec_t         = parametrize_type_over_element_types_and_orders_t< util::UniVector, Element, orders... >;
    using find_result_t       = std::optional< typename el_univec_t::ptr_variant_t >;
    using const_find_result_t = std::optional< typename el_univec_t::const_ptr_variant_t >;

    el_univec_t elements;
    dim_t       dim = uninitialized_dim;
};

template < ElementType ET, el_o_t EO, el_o_t... orders >
void pushToDomain(Domain< orders... >& domain, const Element< ET, EO >& element)
    requires((EO == orders) or ...)
{
    constexpr auto element_dim = ElementTraits< Element< ET, EO > >::native_dim;
    if (domain.dim == Domain< orders... >::uninitialized_dim)
        domain.dim = element_dim;
    util::throwingAssert(domain.dim == element_dim, "Pushing element to domain of differing dimension");
    domain.elements.template getVector< Element< ET, EO > >().push_back(element);
}

template < ElementType ET, el_o_t EO, el_o_t... orders, typename... Args >
void emplaceInDomain(Domain< orders... >& domain, Args&&... args)
    requires((EO == orders) or ...) and std::constructible_from< Element< ET, EO >, Args... >
{
    constexpr auto element_dim = ElementTraits< Element< ET, EO > >::native_dim;
    if (domain.dim == Domain< orders... >::uninitialized_dim)
        domain.dim = element_dim;
    util::throwingAssert(domain.dim == element_dim, "Pushing element to domain of differing dimension");
    domain.elements.template getVector< Element< ET, EO > >().emplace_back(std::forward< Args >(args)...);
}
} // namespace lstr::mesh
#endif // L3STER_MESH_DOMAIN_HPP
