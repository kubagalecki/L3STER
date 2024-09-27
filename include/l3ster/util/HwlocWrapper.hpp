#ifndef L3STER_UTIL_NUMATOPOLOGY_HPP
#define L3STER_UTIL_NUMATOPOLOGY_HPP

#include "l3ster/util/Assertion.hpp"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "hwloc.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace lstr::util::hwloc
{
namespace detail
{
class BitmapWrapper
{
public:
    BitmapWrapper() : m_bitmap{hwloc_bitmap_alloc(), &hwloc_bitmap_free} { assertNotNull(); }
    BitmapWrapper(hwloc_const_bitmap_t bmp) : m_bitmap{hwloc_bitmap_dup(bmp), &hwloc_bitmap_free} { assertNotNull(); }

    operator hwloc_bitmap_t() { return m_bitmap.get(); }
    operator hwloc_const_bitmap_t() const { return m_bitmap.get(); }

    template < std::invocable< unsigned int > Body >
    void forEachSetIndex(Body&& body) const
    {
        unsigned int i{};
        hwloc_bitmap_foreach_begin(i, m_bitmap.get())
        {
            std::invoke(body, i);
        }
        hwloc_bitmap_foreach_end();
    }

private:
    void assertNotNull(std::source_location sl = std::source_location::current()) const
    {
        throwingAssert(bool{m_bitmap}, "Hwloc bitmap allocation failed", sl);
    }

    std::unique_ptr< std::remove_pointer_t< hwloc_bitmap_t >, decltype(&hwloc_bitmap_free) > m_bitmap;
};

inline bool operator==(const BitmapWrapper& b1, const BitmapWrapper& b2)
{
    return hwloc_bitmap_isequal(b1, b2);
}
} // namespace detail

class Topology
{
public:
    Topology() : m_topology{nullptr, &hwloc_topology_destroy}
    {
        init();
        populate();
    }

    size_t getNNodes() const { return m_machine.size(); }
    size_t getNCores(size_t node) const { return m_machine.at(node).second.size(); }
    size_t getNCores() const
    {
        auto node_range = std::views::iota(0uz, getNNodes()) | std::views::common;
        return std::transform_reduce(
            std::ranges::cbegin(node_range), std::ranges::cend(node_range), 0uz, std::plus{}, [this](size_t node) {
                return getNCores(node);
            });
    }
    size_t getNHwThreads(size_t node, size_t core) const
    {
        return static_cast< size_t >(hwloc_bitmap_weight(m_machine.at(node).second.at(core)));
    }
    size_t getNHwThreads() const
    {
        auto node_range = std::views::iota(0uz, getNNodes()) | std::views::common;
        return std::transform_reduce(
            std::ranges::cbegin(node_range), std::ranges::cend(node_range), 0uz, std::plus{}, [this](size_t node) {
                auto core_range = std::views::iota(0uz, getNCores(node)) | std::views::common;
                return std::transform_reduce(std::ranges::cbegin(core_range),
                                             std::ranges::cend(core_range),
                                             0uz,
                                             std::plus{},
                                             [this, node](size_t core) { return getNHwThreads(node, core); });
            });
    }
    bool isEmpty() const { return m_machine.empty(); }

private:
    inline void init();
    inline void populate();

    using Core     = detail::BitmapWrapper;                                   // core cpu set
    using NumaNode = std::pair< detail::BitmapWrapper, std::vector< Core > >; // node set + cores

    std::vector< NumaNode >                                                                         m_machine;
    std::unique_ptr< std::remove_pointer_t< hwloc_topology_t >, decltype(&hwloc_topology_destroy) > m_topology;
};

void Topology::init()
{
    hwloc_topology_t topo_ptr{};
    const auto       init_err = hwloc_topology_init(&topo_ptr);
    throwingAssert(not init_err, "Failed to initialize hwloc topology");
    m_topology.reset(topo_ptr);
    const auto load_err = hwloc_topology_load(m_topology.get());
    throwingAssert(not load_err, "Failed to load hwloc topology");
}

void Topology::populate()
{
    const auto nodeset = detail::BitmapWrapper{hwloc_topology_get_topology_nodeset(m_topology.get())};
    const auto n_nodes = hwloc_bitmap_weight(nodeset);
    m_machine.reserve(static_cast< std::size_t >(n_nodes));
    nodeset.forEachSetIndex([this](unsigned int node) {
        auto& [node_mask, cores] = m_machine.emplace_back();
        hwloc_bitmap_set(node_mask, static_cast< unsigned int >(node));
        auto       node_cpuset       = detail::BitmapWrapper{};
        const auto node_cpu_conv_err = hwloc_cpuset_from_nodeset(m_topology.get(), node_cpuset, node_mask);
        throwingAssert(not node_cpu_conv_err, "Failed to convert hwloc nodeset to cpuset");
        const auto n_cpus_in_node =
            hwloc_get_nbobjs_inside_cpuset_by_type(m_topology.get(), node_cpuset, HWLOC_OBJ_CORE);
        throwingAssert(n_cpus_in_node >= 0, "Failed to determine number of cores in NUMA node");
        const auto n_cpus_in_node_unsigned = static_cast< unsigned int >(n_cpus_in_node);
        cores.reserve(n_cpus_in_node_unsigned);
        for (unsigned int cpu_ind = 0; cpu_ind != n_cpus_in_node_unsigned; ++cpu_ind)
        {
            const auto cpu_ptr =
                hwloc_get_obj_inside_cpuset_by_type(m_topology.get(), node_cpuset, HWLOC_OBJ_CORE, cpu_ind);
            throwingAssert(cpu_ptr, "Failed to query hwloc core object");
            cores.emplace_back(cpu_ptr->cpuset);
        }
    });
}
} // namespace lstr::util::hwloc
#endif // L3STER_UTIL_NUMATOPOLOGY_HPP
