#ifndef L3STER_NUMA_NUMATOPOLOGY_HPP
#define L3STER_NUMA_NUMATOPOLOGY_HPP

#include "hwloc.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace lstr
{
// C++ wrapper for hwloc
// Assumes a symmetric topology, however admits different numbers of PUs at each node
struct HwlocWrapper
{
    inline HwlocWrapper();
    HwlocWrapper(const HwlocWrapper&) = delete;
    HwlocWrapper(HwlocWrapper&&)      = delete;
    HwlocWrapper& operator=(const HwlocWrapper&) = delete;
    HwlocWrapper& operator=(HwlocWrapper&&) = delete;
    inline ~HwlocWrapper();

    [[nodiscard]] size_t         getMachineSize() const noexcept { return cpu_masks.size(); }
    [[nodiscard]] size_t         getNodeSize(size_t node) const noexcept { return cpu_masks[node].size(); }
    [[nodiscard]] const auto&    getNodeMasks(size_t node) const noexcept { return cpu_masks[node]; }
    [[nodiscard]] hwloc_bitmap_t getMachineMask() const noexcept { return hwloc_get_root_obj(topo)->cpuset; }

    inline void                                      bindThreadToCore(size_t node, size_t cpu) const;
    inline void                                      bindThreadToNode(size_t node) const;
    [[nodiscard]] inline std::pair< size_t, size_t > getLastThreadLocation() const;

    [[nodiscard]] inline void* allocateOnNode(size_t size, size_t node) const noexcept;
    void                       free(void* addr, size_t size) const noexcept { hwloc_free(topo, addr, size); }

private:
    inline void initTopology();
    inline void populateTopologyInfo() noexcept;
    inline void groupCpus();

    template < typename F >
    requires std::is_invocable_r_v< bool, F, hwloc_cpuset_t > [[nodiscard]] std::pair< size_t, size_t >
    findCpuIf(const F&) const noexcept;

    std::vector< std::vector< hwloc_cpuset_t > > cpu_masks{};
    std::vector< hwloc_nodeset_t >               node_masks{};
    hwloc_topology_t                             topo{};
};

namespace detail
{
struct HwlocBitmapRaiiWrapper
{
    HwlocBitmapRaiiWrapper() : bmp{hwloc_bitmap_alloc()} {}
    HwlocBitmapRaiiWrapper(const HwlocBitmapRaiiWrapper&) = delete;
    HwlocBitmapRaiiWrapper(HwlocBitmapRaiiWrapper&&)      = delete;
    HwlocBitmapRaiiWrapper& operator=(const HwlocBitmapRaiiWrapper&) = delete;
    HwlocBitmapRaiiWrapper& operator=(HwlocBitmapRaiiWrapper&&) = delete;
    ~HwlocBitmapRaiiWrapper() { hwloc_bitmap_free(bmp); }

    // clang-tidy warns about implicit conversion, but that's exactly the point [NOLINTNEXTLINE]
    operator hwloc_bitmap_t() const { return bmp; }

    hwloc_bitmap_t bmp;
};

template < typename F >
requires std::is_invocable_v< F, hwloc_bitmap_t > void hwlocBitmapForEachWrapper(hwloc_bitmap_t bitmap, const F& fun)
{
    HwlocBitmapRaiiWrapper helper{};
    size_t                 index;
    hwloc_bitmap_foreach_begin(index, bitmap)
    {
        hwloc_bitmap_only(helper, index);
        fun(helper);
    }
    hwloc_bitmap_foreach_end();
}

template < typename F >
requires std::is_invocable_v< F, hwloc_const_bitmap_t > void hwlocBitmapForEachWrapper(hwloc_const_bitmap_t bitmap,
                                                                                       const F&             fun)
{
    HwlocBitmapRaiiWrapper helper{};
    size_t                 index;
    hwloc_bitmap_foreach_begin(index, bitmap)
    {
        hwloc_bitmap_only(helper, index);
        fun(helper);
    }
    hwloc_bitmap_foreach_end();
}
} // namespace detail

inline HwlocWrapper::HwlocWrapper()
{
    initTopology();
    populateTopologyInfo();
    groupCpus();
}

inline HwlocWrapper::~HwlocWrapper()
{
    for (auto& bitmap_vector : cpu_masks)
        std::ranges::for_each(bitmap_vector, [](hwloc_cpuset_t cpu) { hwloc_bitmap_free(cpu); });
    std::ranges::for_each(node_masks, [](hwloc_nodeset_t node) { hwloc_bitmap_free(node); });
    hwloc_topology_destroy(topo);
}

inline void HwlocWrapper::bindThreadToCore(size_t node, size_t cpu) const
{
    if (hwloc_set_cpubind(topo, cpu_masks[node][cpu], HWLOC_CPUBIND_THREAD))
        throw std::runtime_error{"failed not bind thread to core"};
}

inline void HwlocWrapper::bindThreadToNode(size_t node) const
{
    detail::HwlocBitmapRaiiWrapper cpu_set{};
    hwloc_cpuset_from_nodeset(topo, cpu_set, node_masks[node]);
    if (hwloc_set_cpubind(topo, cpu_set, HWLOC_CPUBIND_THREAD))
        throw std::runtime_error{"failed bind thread to NUMA node"};
}

inline std::pair< size_t, size_t > HwlocWrapper::getLastThreadLocation() const
{
    detail::HwlocBitmapRaiiWrapper cpu{};
    if (hwloc_get_last_cpu_location(topo, cpu, HWLOC_CPUBIND_THREAD))
        throw std::runtime_error("hwloc failed to obtain the last location of the current thread");
    const auto ret = findCpuIf([&](hwloc_const_cpuset_t set) { return hwloc_bitmap_isequal(set, cpu); });
    if (ret.first == std::numeric_limits< size_t >::max())
        throw std::logic_error{"The thread location provided by hwloc seems to lie outside of the machine's topology"};
    return ret;
}

inline void* HwlocWrapper::allocateOnNode(size_t size, size_t node) const noexcept
{
    return hwloc_alloc_membind(topo, size, node_masks[node], HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
}

inline void HwlocWrapper::initTopology()
{
    if (hwloc_topology_init(&topo) || hwloc_topology_load(topo))
        throw std::runtime_error{"hwloc failed to obtain topology information"};
}

inline void HwlocWrapper::populateTopologyInfo() noexcept
{
    hwloc_const_nodeset_t node_set = hwloc_topology_get_topology_nodeset(topo);
    const size_t          n_nodes  = hwloc_bitmap_weight(node_set);
    cpu_masks.reserve(n_nodes);
    node_masks.reserve(n_nodes);
    detail::HwlocBitmapRaiiWrapper cpu_set{};
    detail::hwlocBitmapForEachWrapper(node_set, [&](hwloc_const_nodeset_t node) {
        node_masks.push_back(hwloc_bitmap_dup(node));
        hwloc_cpuset_from_nodeset(topo, cpu_set, node);
        std::vector< hwloc_cpuset_t > cpus;
        cpus.reserve(hwloc_bitmap_weight(cpu_set));
        detail::hwlocBitmapForEachWrapper(cpu_set, [&](hwloc_cpuset_t cpu) { cpus.push_back(hwloc_bitmap_dup(cpu)); });
        cpu_masks.push_back(std::move(cpus));
    });
}

inline void HwlocWrapper::groupCpus()
{
    const auto cpu_to_obj = [&](hwloc_cpuset_t set) {
        return hwloc_get_obj_inside_cpuset_by_type(topo, set, HWLOC_OBJ_PU, 0);
    };
    const auto pu_depth = hwloc_get_type_depth(topo, HWLOC_OBJ_PU);
    const auto cpu_dist = [&](hwloc_cpuset_t cpu1, hwloc_cpuset_t cpu2) {
        return pu_depth - hwloc_get_common_ancestor_obj(topo, cpu_to_obj(cpu1), cpu_to_obj(cpu2))->depth;
    };
    for (auto& node : cpu_masks)
    {
        if (node.size() < 2)
            continue;
        for (auto it = begin(node); it + 2 != end(node); ++it)
            std::partial_sort(it + 1, it + 2, end(node), [&](hwloc_cpuset_t c1, hwloc_cpuset_t c2) {
                return cpu_dist(*it, c1) < cpu_dist(*it, c2);
            });
    }
}

template < typename F >
requires std::is_invocable_r_v< bool, F, hwloc_cpuset_t > std::pair< size_t, size_t >
                                                          HwlocWrapper::findCpuIf(const F& fun) const noexcept
{
    std::pair< size_t, size_t > ret;
    const size_t                node_dist = std::distance(
        cpu_masks.begin(), std::ranges::find_if(cpu_masks, [&](const std::vector< hwloc_cpuset_t >& cpus) {
            ret.second = std::distance(cpus.begin(), std::ranges::find_if(cpus, fun));
            return ret.second < cpus.size();
        }));
    ret.first = node_dist == cpu_masks.size() ? std::numeric_limits< size_t >::max() : node_dist;
    return ret;
}
} // namespace lstr

#endif // L3STER_NUMA_NUMATOPOLOGY_HPP