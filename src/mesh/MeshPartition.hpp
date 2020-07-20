#ifndef L3STER_MESH_MESHPARTITION_HPP
#define L3STER_MESH_MESHPARTITION_HPP

#include <map>
#include <variant>

namespace lstr::mesh
{
class MeshPartition
{
public:
    using domain_map_t = std::map< types::d_id_t, Domain >;

    MeshPartition()                     = default;
    MeshPartition(const MeshPartition&) = delete;
    MeshPartition(MeshPartition&&)      = default;
    MeshPartition& operator=(const MeshPartition&) = delete;
    MeshPartition& operator=(MeshPartition&&) = default;

    explicit MeshPartition(MeshPartition::domain_map_t&& domains_) : domains(domains_) {}

    // Visit domains
    template < typename F >
    auto visitAllElements(F&& element_visitor);

    template < typename F >
    auto cvisitAllElements(F&& element_visitor) const;

    template < typename F >
    auto visitSpecifiedDomains(F&& element_visitor, const std::vector< types::d_id_t >& domain_ids);

    template < typename F >
    auto cvisitSpecifiedDomains(F&&                                 element_visitor,
                                const std::vector< types::d_id_t >& domain_ids) const;

    template < typename F, typename D >
    auto visitDomainIf(F&& element_visitor, D&& domain_predicate);

    template < typename F, typename D >
    auto cvisitDomainIf(F&& element_visitor, D&& domain_predicate) const;

    // Find element
    // Note: Elements are not ordered; if predicate returns true for multiple elements, the
    // reference to any one of them may be returned
    template < typename F >
    std::optional< element_ref_variant_t > findElement(const F& predicate);

    template < typename F >
    std::optional< element_ref_variant_t >
    findElementInSpecifiedDomains(const F&                            predicate,
                                  const std::vector< types::d_id_t >& domain_ids);

    template < typename F, typename D >
    std::optional< element_ref_variant_t > findElementIfDomain(const F& predicate,
                                                               const D& domain_predicate);

    inline void pushDomain(types::d_id_t, Domain&&);
    inline void pushDomain(types::d_id_t, const Domain&);
    inline void popDomain(types::d_id_t);

private:
    domain_map_t domains;
};

template < typename F >
auto MeshPartition::visitAllElements(F&& element_visitor)
{
    return visitDomainIf(std::forward< F >(element_visitor),
                         [](const DomainView&) { return true; });
}

template < typename F >
auto MeshPartition::cvisitAllElements(F&& element_visitor) const
{
    return cvisitDomainIf(std::forward< F >(element_visitor),
                          [](const DomainView&) { return true; });
}

template < typename F >
auto MeshPartition::visitSpecifiedDomains(F&&                                 element_visitor,
                                          const std::vector< types::d_id_t >& domain_ids)
{
    auto domain_predicate = [&domain_ids](const DomainView& d) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&d](const types::d_id_t& d1) {
            return d.getID() == d1;
        });
    };

    return visitDomainIf(std::forward< F >(element_visitor), std::move(domain_predicate));
}

template < typename F >
auto MeshPartition::cvisitSpecifiedDomains(F&&                                 element_visitor,
                                           const std::vector< types::d_id_t >& domain_ids) const
{
    auto domain_predicate = [&domain_ids](const DomainView& d) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&d](const types::d_id_t& d1) {
            return d.getID() == d1;
        });
    };

    return cvisitDomainIf(std::forward< F >(element_visitor), std::move(domain_predicate));
}

template < typename F, typename D >
auto MeshPartition::visitDomainIf(F&& element_visitor, D&& domain_predicate)
{
    static_assert(std::is_invocable_v< D, DomainView >);

    auto visitor   = std::forward< F >(element_visitor);
    auto predicate = std::forward< D >(domain_predicate);

    const auto domain_visitor = [&visitor](domain_map_t::value_type& domain) {
        domain.second.visit(visitor);
    };

    std::for_each(domains.begin(),
                  domains.end(),
                  [&domain_visitor, &predicate](domain_map_t::value_type& domain) {
                      if (predicate(DomainView{domain.second, domain.first}))
                          domain_visitor(domain);
                  });

    return visitor;
}

template < typename F, typename D >
auto MeshPartition::cvisitDomainIf(F&& element_visitor, D&& domain_predicate) const
{
    static_assert(std::is_invocable_v< D, DomainView >);

    auto visitor   = std::forward< F >(element_visitor);
    auto predicate = std::forward< D >(domain_predicate);

    const auto domain_visitor = [&visitor](const domain_map_t::value_type& domain) {
        domain.second.cvisit(visitor);
    };

    std::for_each(domains.cbegin(),
                  domains.cend(),
                  [&domain_visitor, &predicate](const domain_map_t::value_type& domain) {
                      if (predicate(DomainView{domain.second, domain.first}))
                          domain_visitor(domain);
                  });

    return std::move(visitor);
}

template < typename F >
std::optional< element_ref_variant_t > MeshPartition::findElement(const F& predicate)
{
    return findElementIfDomain(predicate, [](const auto&) { return true; });
}

template < typename F >
std::optional< element_ref_variant_t >
MeshPartition::findElementInSpecifiedDomains(const F&                            predicate,
                                             const std::vector< types::d_id_t >& domain_ids)
{
    return findElementIfDomain(predicate, [&domain_ids](const auto& domain_view) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&domain_view](types::d_id_t d) {
            return d == domain_view.getID();
        });
    });
}

template < typename F, typename D >
std::optional< element_ref_variant_t > MeshPartition::findElementIfDomain(const F& predicate,
                                                                          const D& domain_predicate)
{
    std::optional< element_ref_variant_t > ret_val;

    for (auto& domain_map_entry : domains)
    {
        if (domain_predicate(DomainView{domain_map_entry.second, domain_map_entry.first}))
        {
            ret_val = domain_map_entry.second.findElement(predicate);

            if (ret_val)
                break;
        }
    };

    return ret_val;
}

void MeshPartition::pushDomain(types::d_id_t id, Domain&& d)
{
    domains[id] = std::move(d);
}

void MeshPartition::pushDomain(types::d_id_t id, const Domain& d)
{
    domains[id] = d;
}

void MeshPartition::popDomain(types::d_id_t id)
{
    domains.erase(id);
}
} // namespace lstr::mesh

#endif // L3STER_MESH_MESHPARTITION_HPP
