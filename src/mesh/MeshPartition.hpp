// Top level data structure for storing the entire mesh

#ifndef L3STER_INCGUARD_MESH_MESHPARTITION_HPP
#define L3STER_INCGUARD_MESH_MESHPARTITION_HPP

#include "definitions/Typedefs.h"
#include "mesh/Domain.hpp"

#include <map>
#include <memory_resource>
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
    template < typename F >
    void visitAllElements(F&&);

    template < typename F >
    void cvisitAllElements(F&&) const;

    template < typename F, typename D >
    void visitDomainIf(F&& fun, D&& domain_predicate);

    template < typename F, typename D >
    void cvisitDomainIf(F&& fun, D&& domain_predicate) const;

    inline void pushDomain(types::d_id_t, Domain&&);
    inline void pushDomain(types::d_id_t, const Domain&);
    inline void popDomain(types::d_id_t);

private:
    // Domains are kept in a map allocated to a buffer using the STL PMR to avoid memory
    // fragmentation. This is merely an optimization, once the buffer is exhausted the allocator
    // will fall back to heap allocation (vide documentation of the <memory_resource> header)
    constexpr static inline size_t      domain_buffer_size = 256;
    std::byte                           domain_buffer[domain_buffer_size];
    std::pmr::monotonic_buffer_resource domain_memory_resource{domain_buffer, domain_buffer_size};

    using domain_map_t = std::pmr::map< types::d_id_t, Domain >;
    domain_map_t domains{&domain_memory_resource};
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
