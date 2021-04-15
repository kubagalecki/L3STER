#ifndef L3STER_UTIL_METISUTILS_HPP
#define L3STER_UTIL_METISUTILS_HPP
#include "metis.h"

#include "mesh/Mesh.hpp"

#include <memory>

namespace lstr::detail
{
inline void handleMetisErrorCode(int error)
{
    switch (error)
    {
    case METIS_OK:
        break;
    case METIS_ERROR_MEMORY:
        throw std::bad_alloc{};
    default:
        throw std::runtime_error{"Metis failed to partition the mesh"};
    }
}

inline auto convertMeshToMetisFormat(const MeshPartition& partition)
{
    std::vector< idx_t > eind, eptr;
    size_t               topo_size = 0;
    partition.cvisit([&](const auto& element) { topo_size += element.getNodes().size(); });
    eind.reserve(topo_size);
    eptr.reserve(partition.getNElements() + 1);
    eptr.push_back(0);
    partition.cvisit([&](const auto& element) {
        std::ranges::for_each(element.getNodes(), [&](auto n) { eind.push_back(n); });
        eptr.push_back(element.getNodes().size());
    });
    return std::make_pair(std::move(eptr), std::move(eind));
}

inline auto getMeshDualGraph(const MeshPartition& part)
{
    auto mesh_in_metis_format = detail::convertMeshToMetisFormat(part);
    auto& [eptr, eind]        = mesh_in_metis_format;

    auto  ne      = static_cast< idx_t >(part.getNElements());
    auto  nn      = static_cast< idx_t >(part.getNodes().size());
    idx_t ncommon = 2;
    idx_t numflag = 0;

    idx_t* xadj;
    idx_t* adjncy;

    const auto error = METIS_MeshToDual(&ne, &nn, eptr.data(), eind.data(), &ncommon, &numflag, &xadj, &adjncy);
    detail::handleMetisErrorCode(error);

    constexpr auto deleter = [](idx_t* ptr) {
        METIS_Free(ptr);
    };
    std::unique_ptr< idx_t[], decltype(deleter) > E_inds{xadj, deleter}, E{adjncy, deleter};
    return std::make_pair(std::move(E_inds), std::move(E));
}
} // namespace lstr::detail
#endif // L3STER_UTIL_METISUTILS_HPP
