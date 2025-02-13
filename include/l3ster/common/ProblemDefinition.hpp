#ifndef L3STER_DOFS_PROBLEMDEFINITION_HPP
#define L3STER_DOFS_PROBLEMDEFINITION_HPP

#include "l3ster/common/Typedefs.h"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/ArrayOwner.hpp"

#include <map>

namespace lstr
{
template < size_t fields >
struct DomainDef
{
    d_id_t                     domain{};
    std::array< bool, fields > active_fields{};
};

namespace detail
{
struct AllDofsTag
{};
} // namespace detail
inline constexpr detail::AllDofsTag ALL_DOFS{};

template < size_t fields >
constexpr auto defineDomain(d_id_t domain, std::convertible_to< size_t > auto... field_inds) -> DomainDef< fields >
{
    auto active_fields = std::array< bool, fields >{};
    (
        [&](auto ind) {
            active_fields.at(static_cast< size_t >(ind)) = true;
        }(field_inds),
        ...);
    return {domain, active_fields};
}

template < size_t fields >
constexpr auto defineDomain(d_id_t domain, detail::AllDofsTag) -> DomainDef< fields >
{
    auto active_fields = std::array< bool, fields >{};
    active_fields.fill(true);
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

template < size_t total_num_dofs >
class ProblemDefinition
{
public:
    static constexpr auto max_dofs_per_node = total_num_dofs;
    struct DomainDefinition
    {
        util::ArrayOwner< d_id_t >    domains;
        std::bitset< total_num_dofs > active_dofs;
    };

    ProblemDefinition() = default;
    ProblemDefinition(util::ArrayOwner< d_id_t > domains) { define(std::move(domains)); }

    void define(util::ArrayOwner< d_id_t >        domains,
                const util::ArrayOwner< size_t >& inds = util::makeIotaArray< size_t, total_num_dofs >())
    {
        util::throwingAssert(std::ranges::all_of(inds, [](auto i) { return i < total_num_dofs; }),
                             "Out of bounds active unknown index");
        auto active_dofs = std::bitset< total_num_dofs >{};
        for (auto i : inds)
            active_dofs.set(i);
        m_domain_defs.push_back(DomainDefinition{std::move(domains), active_dofs});
    }

    [[nodiscard]] auto begin() const { return m_domain_defs.begin(); }
    [[nodiscard]] auto end() const { return m_domain_defs.end(); }
    [[nodiscard]] auto size() const { return m_domain_defs.size(); }
    [[nodiscard]] bool empty() const { return m_domain_defs.empty(); }

private:
    std::vector< DomainDefinition > m_domain_defs;
};

namespace detail
{
template < size_t num_domains, size_t num_dofs >
auto convertToRuntime(const ProblemDef< num_domains, num_dofs >& problem_def) -> ProblemDefinition< num_dofs >
{
    auto inverse = std::map< std::bitset< num_dofs >, std::vector< d_id_t > >{};
    for (const auto& [domain, dof_bmp] : problem_def)
        inverse[util::toStdBitset(dof_bmp)].push_back(domain);
    auto retval = ProblemDefinition< num_dofs >{};
    for (const auto& [dof_bmp, domains] : inverse)
        retval.define(domains, dof_bmp);
    return retval;
}
} // namespace detail
} // namespace lstr
#endif // L3STER_DOFS_PROBLEMDEFINITION_HPP
