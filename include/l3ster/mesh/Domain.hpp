#ifndef L3STER_MESH_DOMAIN_HPP
#define L3STER_MESH_DOMAIN_HPP

#include "l3ster/common/Enums.hpp"
#include "l3ster/mesh/Element.hpp"
#include "l3ster/mesh/ElementMeta.hpp"
#include "l3ster/util/Common.hpp"
#include "l3ster/util/UniVector.hpp"

namespace lstr::mesh
{
template < el_o_t... orders >
struct Domain
{
    static_assert(sizeof...(orders) > 0);

    static constexpr auto uninitialized_dim = std::numeric_limits< dim_t >::max();

    using el_univec_t = parametrize_type_over_element_types_and_orders_t< util::UniVector, Element, orders... >;

    [[nodiscard]] inline size_t numElements() const;

    el_univec_t elements;
    dim_t       dim = uninitialized_dim;
};

template < el_o_t... orders >
size_t Domain< orders... >::numElements() const
{
    const auto sizes = elements.sizes();
    return std::reduce(sizes.begin(), sizes.end());
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

template < ElementType ET, el_o_t EO, el_o_t... orders >
void pushToDomain(Domain< orders... >& domain, const Element< ET, EO >& element)
    requires((EO == orders) or ...)
{
    emplaceInDomain< ET, EO >(domain, element);
}
} // namespace lstr::mesh
#endif // L3STER_MESH_DOMAIN_HPP
