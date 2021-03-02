#ifndef L3STER_NUMA_NUMATOPOLOGY_HPP
#define L3STER_NUMA_NUMATOPOLOGY_HPP

#include "hwloc.h"

#include <stdexcept>
#include <vector>

namespace lstr
{
// C++ wrapper for hwloc
// Assumes all NUMA nodes and PUs are at the same depths, however admits different numbers of PUs at each node
// Assumes NUMA nodes are equivalent to hwloc "packages"
struct TopologyWrapper
{
    inline TopologyWrapper();
    TopologyWrapper(const TopologyWrapper&) = delete;
    TopologyWrapper(TopologyWrapper&&)      = delete;
    TopologyWrapper& operator=(const TopologyWrapper&) = delete;
    TopologyWrapper& operator=(TopologyWrapper&&) = delete;
    ~TopologyWrapper() { hwloc_topology_destroy(topo); }

    [[nodiscard]] size_t         getMachineSize() const noexcept { return mask_map.size(); }
    [[nodiscard]] size_t         getNodeSize(size_t node) const noexcept { return mask_map[node].size(); }
    [[nodiscard]] const auto&    getNodeMasks(size_t node) const noexcept { return mask_map[node]; }
    [[nodiscard]] hwloc_bitmap_t getMachineMask() const noexcept { return hwloc_get_root_obj(topo)->cpuset; }

    void                                      bindThread(size_t node, size_t cpu) const;
    [[nodiscard]] std::pair< size_t, size_t > getLastThreadLocation() const;

private:
    hwloc_topology_t                             topo{};
    std::vector< std::vector< hwloc_cpuset_t > > mask_map{};
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
void lookup_pus_recursive(hwloc_topology_t topology, hwloc_obj_t obj, const F& f)
{
    const int pu_depth = hwloc_topology_get_depth(topology) - 1;
    if (obj->depth < pu_depth)
    {
        for (size_t i = 0; i < obj->arity; ++i)
            lookup_pus_recursive(topology, obj->children[i], f);
    }
    else
        f(obj);
}
} // namespace detail
inline TopologyWrapper::TopologyWrapper()
{
    int status = hwloc_topology_init(&topo);
    if (status)
        throw std::runtime_error{"hwloc failed to initialize"};
    status = hwloc_topology_load(topo);
    if (status)
        throw std::runtime_error{"hwloc failed to load the node topology"};
    const size_t n_numa_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PACKAGE);
    const size_t n_pu         = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PU);
    mask_map.resize(n_numa_nodes);
    for (hwloc_obj_t node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PACKAGE, 0); auto& node_pus : mask_map)
    {
        node_pus.reserve(n_pu / n_numa_nodes);
        // DFS node branch of topology tree for PUs
        detail::lookup_pus_recursive(topo, node, [&](hwloc_obj_t obj) { node_pus.push_back(obj->cpuset); });
        node_pus.shrink_to_fit();
        node = hwloc_get_next_obj_by_type(topo, HWLOC_OBJ_PACKAGE, node);
    }
}

void TopologyWrapper::bindThread(size_t node, size_t cpu) const
{
    const int status = hwloc_set_cpubind(topo, mask_map[node][cpu], HWLOC_CPUBIND_THREAD);
    if (status)
        throw std::runtime_error{"could not bind thread to core"};
}

std::pair< size_t, size_t > TopologyWrapper::getLastThreadLocation() const
{
    detail::HwlocBitmapRaiiWrapper cpu{};
    const int                      status = hwloc_get_last_cpu_location(topo, cpu, HWLOC_CPUBIND_THREAD);
    if (status)
        throw std::runtime_error("hwloc failed to obtain the last location of the current thread");
    for (size_t node_index = 0; const auto& node : mask_map)
    {
        const auto it = std::ranges::find_if(node, [&](hwloc_cpuset_t set) { return hwloc_bitmap_isequal(set, cpu); });
        if (it != node.end())
            return std::pair< size_t, size_t >{node_index, std::distance(node.begin(), it)};
        ++node_index;
    }
    throw std::logic_error{"The thread location provided by hwloc seems to lie outside of the machine's topology"};
}
} // namespace lstr

#endif // L3STER_NUMA_NUMATOPOLOGY_HPP
