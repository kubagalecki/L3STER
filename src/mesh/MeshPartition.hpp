// Top level data structure for storing the entire mesh

#ifndef L3STER_INCGUARD_MESH_MESHPARTITION_HPP
#define L3STER_INCGUARD_MESH_MESHPARTITION_HPP

#include "definitions/Typedefs.h"
#include "mesh/Domain.hpp"

#include <map>
#include <variant>

namespace lstr::mesh
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                     MESHPARTITION CLASS                                  //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
MeshPartition class - a mapping of IDs to domains
*/
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

    template < typename F >
    void visitAllElements(F&&);

    template < typename F >
    void cvisitAllElements(F&&) const;

    template < typename F, size_t N >
    void visitSpecifiedDomains(F&& fun, const std::array< types::d_id_t, N >& domain_ids);

    template < typename F, size_t N >
    void cvisitSpecifiedDomains(F&& fun, const std::array< types::d_id_t, N >& domain_ids) const;

    template < typename F, typename D >
    void visitDomainIf(F&& fun, D&& domain_predicate);

    template < typename F, typename D >
    void cvisitDomainIf(F&& fun, D&& domain_predicate) const;

    inline void pushDomain(types::d_id_t, Domain&&);
    inline void pushDomain(types::d_id_t, const Domain&);
    inline void popDomain(types::d_id_t);

private:
    domain_map_t domains;
};

template < typename F >
void MeshPartition::visitAllElements(F&& fun)
{
    const auto element_vector_visitor = [v = std::forward< F >(fun)](auto& el_v) {
        std::for_each(el_v.begin(), el_v.end(), v);
    };
    std::for_each(domains.begin(), domains.end(), [&](domain_map_t::value_type& d) {
        d.second.visit(element_vector_visitor);
    });
}

template < typename F >
void MeshPartition::cvisitAllElements(F&& fun) const
{
    const auto element_vector_visitor = [v = std::forward< F >(fun)](const auto& el_v) {
        std::for_each(el_v.cbegin(), el_v.cend(), v);
    };
    std::for_each(domains.cbegin(), domains.cend(), [&](const domain_map_t::value_type& d) {
        d.second.cvisit(element_vector_visitor);
    });
}

template < typename F, size_t N >
void MeshPartition::visitSpecifiedDomains(F&& fun, const std::array< types::d_id_t, N >& domain_ids)
{
    const auto domain_predicate = [&domain_ids](const types::d_id_t& d) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&d](const types::d_id_t& d1) {
            return d == d1;
        });
    };

    visitDomainIf(std::forward< F >(fun), domain_predicate);
}

template < typename F, size_t N >
void MeshPartition::cvisitSpecifiedDomains(F&&                                   fun,
                                           const std::array< types::d_id_t, N >& domain_ids) const
{
    const auto domain_predicate = [&domain_ids](const types::d_id_t& d) {
        return std::any_of(domain_ids.cbegin(), domain_ids.cend(), [&d](const types::d_id_t& d1) {
            return d == d1;
        });
    };

    cvisitDomainIf(std::forward< F >(fun), domain_predicate);
}

template < typename F, typename D >
void MeshPartition::visitDomainIf(F&& fun, D&& domain_predicate)
{
    const auto element_vector_visitor = [v = std::forward< F >(fun)](auto& el_v) {
        std::for_each(el_v.begin(), el_v.end(), v);
    };
    std::for_each(domains.begin(),
                  domains.end(),
                  [&element_vector_visitor,
                   dp = std::forward< D >(domain_predicate)](domain_map_t::value_type& d) {
                      if (dp(d.first))
                          d.second.visit(element_vector_visitor);
                  });
}

template < typename F, typename D >
void MeshPartition::cvisitDomainIf(F&& fun, D&& domain_predicate) const
{
    const auto element_vector_visitor = [v = std::forward< F >(fun)](const auto& el_v) {
        std::for_each(el_v.cbegin(), el_v.cend(), v);
    };
    std::for_each(domains.cbegin(),
                  domains.cend(),
                  [&element_vector_visitor,
                   dp = std::forward< D >(domain_predicate)](const domain_map_t::value_type& d) {
                      if (dp(d.first))
                          d.second.cvisit(element_vector_visitor);
                  });
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

#endif // L3STER_INCGUARD_MESH_MESHPARTITION_HPP
