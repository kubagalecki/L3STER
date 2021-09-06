#ifndef L3STER_NUMA_NODERESOURCEMANAGER_HPP
#define L3STER_NUMA_NODERESOURCEMANAGER_HPP

#include "global_resource/MpiComm.hpp"
#include "global_resource/NodeGlobalResource.hpp"

namespace lstr
{
class NodeResourceManager
{
private:
    HwlocWrapper                      topo_ptr;
    std::vector< NodeGlobalResource > memory_resources;
    std::vector< MpiComm >            comms;
};
} // namespace lstr
#endif // L3STER_NUMA_NODERESOURCEMANAGER_HPP
