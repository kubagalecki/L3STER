#ifndef L3STER_ALGSYS_OPERATORUTILS_HPP
#define L3STER_ALGSYS_OPERATORUTILS_HPP

#include "l3ster/mesh/LocalMeshView.hpp"

namespace lstr::algsys::detail
{
template < el_o_t... orders >
bool checkDomainDimension(const mesh::MeshPartition< orders... >& mesh,
                          const util::ArrayOwner< d_id_t >&       ids,
                          d_id_t                                  dim)
{
    const auto check_domain_dim = [&](d_id_t id) {
        try
        {
            const auto domain_dim = mesh.getDomain(id).dim;
            return domain_dim == dim;
        }
        catch (const std::out_of_range&) // Domain not present in partition means kernel will not be invoked
        {
            return true;
        }
    };
    return std::ranges::all_of(ids, check_domain_dim);
}

template < el_o_t... orders >
bool checkDomainDimension(const mesh::LocalMeshView< orders... >& mesh,
                          const util::ArrayOwner< d_id_t >&       ids,
                          d_id_t                                  dim)
{
    const auto check_domain_dim = [&](d_id_t id) {
        try
        {
            const auto domain_dim = mesh.getDomain(id).dim;
            return domain_dim == dim;
        }
        catch (const std::out_of_range&) // Domain not present in partition means kernel will not be invoked
        {
            return true;
        }
    };
    return std::ranges::all_of(ids, check_domain_dim);
}
} // namespace lstr::algsys::detail
#endif // L3STER_ALGSYS_OPERATORUTILS_HPP
