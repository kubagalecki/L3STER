#ifndef L3STER_NUMA_NODEGLOBAL_RESOURCE_HPP
#define L3STER_NUMA_NODEGLOBAL_RESOURCE_HPP

#include "numa/HwlocWrapper.hpp"

#include <memory>
#include <memory_resource>
#include <unordered_map>

namespace lstr
{
class NodeGlobalResource final : public std::pmr::memory_resource
{
public:
    explicit NodeGlobalResource(HwlocWrapper* topo_ptr_, size_t node_) : topo_ptr{topo_ptr_}, node{node_} {}
    NodeGlobalResource(const NodeGlobalResource&) = delete;
    NodeGlobalResource(NodeGlobalResource&&)      = delete;
    NodeGlobalResource& operator=(const NodeGlobalResource&) = delete;
    NodeGlobalResource& operator=(NodeGlobalResource&&) = delete;
    ~NodeGlobalResource() final                         = default;

    [[nodiscard]] inline void* do_allocate(size_t bytes, size_t alignment) final;
    inline void                do_deallocate(void* p, size_t bytes, size_t alignment) final;
    [[nodiscard]] inline bool  do_is_equal(const memory_resource& other) const noexcept final;

private:
    HwlocWrapper*                           topo_ptr;
    std::pmr::unsynchronized_pool_resource  internal_resource{};
    std::pmr::unordered_map< void*, void* > alloc_map{&internal_resource};
    size_t                                  node;
};

void* NodeGlobalResource::do_allocate(size_t bytes, size_t alignment)
{
    // Allocate
    const size_t alloc_size     = bytes + alignment - 1;
    void*        alloc_location = topo_ptr->allocateOnNode(alloc_size, node);
    if (!alloc_location)
        throw std::bad_alloc{};

    // Align
    size_t throwaway        = std::numeric_limits< size_t >::max();
    void*  aligned_location = alloc_location;
    std::align(alignment, alloc_size, aligned_location, throwaway); // guaranteed to succeed

    // Pointer bookkeeping for deallocation
    alloc_map[aligned_location] = alloc_location;

    return aligned_location;
}

void NodeGlobalResource::do_deallocate(void* p, size_t bytes, size_t alignment)
{
    const auto it = alloc_map.find(p);
    topo_ptr->free(it->second, bytes + alignment - 1);
    alloc_map.erase(it);
}

bool NodeGlobalResource::do_is_equal(const std::pmr::memory_resource& other) const noexcept
{
    return this == &other;
}
} // namespace lstr
#endif // L3STER_NUMA_NODEGLOBAL_RESOURCE_HPP
