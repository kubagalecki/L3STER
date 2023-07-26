#ifndef L3STER_DOFS_PROBLEMDEFINITION_HPP
#define L3STER_DOFS_PROBLEMDEFINITION_HPP

#include "l3ster/common/Typedefs.h"

#include <algorithm>
#include <array>
#include <concepts>

namespace lstr
{
template < size_t fields >
struct DomainDef
{
    d_id_t                     domain{};
    std::array< bool, fields > active_fields{};
};

template < size_t fields >
constexpr auto defineDomain(d_id_t domain, std::convertible_to< size_t > auto... field_inds) -> DomainDef< fields >
{
    auto                        active_fields = std::array< bool, fields >{};
    [[maybe_unused]] const auto set           = [&](auto ind) {
        active_fields.at(static_cast< size_t >(ind)) = true;
    };
    (set(field_inds), ...);
    return {domain, active_fields};
}

namespace detail
{
template < std::integral T >
constexpr auto getNFields(std::initializer_list< T > list) -> size_t
{
    return static_cast< size_t >(std::max(list) + 1);
}
constexpr auto getNFields() -> size_t
{
    return 0;
}
} // namespace detail
#define L3STER_DEFINE_DOMAIN(domain, ...)                                                                              \
    defineDomain< ::lstr::detail::getNFields(__VA_OPT__({) __VA_ARGS__ __VA_OPT__(})) >(domain __VA_OPT__(, ) __VA_ARGS__)

template < size_t domains, size_t fields >
class ProblemDef
{
public:
    static constexpr size_t n_domains = domains;
    static constexpr size_t n_fields  = fields;

    constexpr ProblemDef() = default;
    template < size_t... comp_fields >
    constexpr explicit ProblemDef(const DomainDef< comp_fields >&... domain_defs_)
        : domain_defs{extendDomainDef(domain_defs_)...}
    {}

    constexpr auto begin() const { return domain_defs.cbegin(); }
    constexpr auto end() const { return domain_defs.cend(); }
    constexpr auto size() const { return domain_defs.size(); }

    std::array< DomainDef< fields >, domains > domain_defs{};

private:
    template < size_t nf >
    constexpr auto extendDomainDef(const DomainDef< nf >& dom_def) -> DomainDef< fields >
        requires(nf <= fields)
    {
        auto retval = DomainDef< fields >{};
        retval.active_fields.fill(false);
        retval.domain = dom_def.domain;
        std::ranges::copy(dom_def.active_fields, retval.active_fields.begin());
        return retval;
    }
};
template < size_t... fields >
ProblemDef(const DomainDef< fields >&... dom_defs) -> ProblemDef< sizeof...(dom_defs), std::max({fields...}) >;

using EmptyProblemDef = ProblemDef< 0, 0 >;
} // namespace lstr
#endif // L3STER_DOFS_PROBLEMDEFINITION_HPP
