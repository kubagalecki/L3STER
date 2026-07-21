#ifndef L3STER_UTIL_NUMATOPOLOGY_HPP
#define L3STER_UTIL_NUMATOPOLOGY_HPP

#include "l3ster/util/Assertion.hpp"

#include "hwloc.h"

#include <functional>
#include <memory>

namespace lstr::util::hwloc
{
class Mask
{
public:
    Mask() : m_bitmap{hwloc_bitmap_alloc(), &hwloc_bitmap_free} { assertNotNull(); }
    Mask(hwloc_const_bitmap_t bmp) : m_bitmap{hwloc_bitmap_dup(bmp), &hwloc_bitmap_free} { assertNotNull(); }

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

    void set(unsigned index) { throwingAssert(not hwloc_bitmap_set(m_bitmap.get(), index)); }
    bool get(unsigned index) const { return hwloc_bitmap_isset(m_bitmap.get(), index); }
    auto weight() const -> unsigned { return static_cast< unsigned >(hwloc_bitmap_weight(m_bitmap.get())); }

    auto operator!() const -> Mask
    {
        auto retval = Mask{};
        hwloc_bitmap_not(retval, m_bitmap.get());
        return retval;
    }

private:
    void assertNotNull(std::source_location sl = std::source_location::current()) const
    {
        throwingAssert(bool{m_bitmap}, "Hwloc bitmap allocation failed", sl);
    }

    std::unique_ptr< std::remove_pointer_t< hwloc_bitmap_t >, decltype(&hwloc_bitmap_free) > m_bitmap;
};

inline bool operator==(const Mask& b1, const Mask& b2)
{
    return hwloc_bitmap_isequal(b1, b2);
}

#define L3STER_DEFINE_HWLOC_BMP_BINOP(symbol, op)                                                                      \
    inline Mask operator symbol(const Mask& b1, const Mask& b2)                                                        \
    {                                                                                                                  \
        auto retval = Mask{};                                                                                          \
        op(retval, b1, b2);                                                                                            \
        return retval;                                                                                                 \
    }
L3STER_DEFINE_HWLOC_BMP_BINOP(&, hwloc_bitmap_and);
L3STER_DEFINE_HWLOC_BMP_BINOP(|, hwloc_bitmap_or);
L3STER_DEFINE_HWLOC_BMP_BINOP(^, hwloc_bitmap_xor);
#undef L3STER_DEFINE_HWLOC_BMP_BINOP

class Topology
{
public:
    inline Topology();

    inline auto getNNodes() const -> unsigned;
    inline auto getNCores() const -> unsigned;
    inline auto getNHwThreads() const -> unsigned;

    auto        getFullMask() const -> Mask { return Mask{hwloc_topology_get_topology_cpuset(m_topology.get())}; }
    inline auto getNodeMask(unsigned node) const -> Mask;
    inline auto getCoreMask(unsigned core) const -> Mask;

    inline void bindThread(const Mask& cpu_mask) const;
    inline auto getThreadBinding() const -> Mask;

private:
    using topology_t = std::remove_pointer_t< hwloc_topology_t >;

    std::unique_ptr< topology_t, decltype(&hwloc_topology_destroy) > m_topology;
};

Topology::Topology() : m_topology{nullptr, &hwloc_topology_destroy}
{
    const auto init_err = hwloc_topology_init(std::out_ptr(m_topology));
    throwingAssert(not init_err, "Failed to initialize hwloc topology");
    const auto load_err = hwloc_topology_load(m_topology.get());
    throwingAssert(not load_err, "Failed to load hwloc topology");
}
auto Topology::getNNodes() const -> unsigned
{
    const auto num_nodes = hwloc_get_nbobjs_by_type(m_topology.get(), HWLOC_OBJ_NUMANODE);
    throwingAssert(num_nodes > 0, "Failed to get number of NUMA nodes from hwloc");
    return static_cast< unsigned >(num_nodes);
}

auto Topology::getNCores() const -> unsigned
{
    const auto num_cores = hwloc_get_nbobjs_by_type(m_topology.get(), HWLOC_OBJ_CORE);
    throwingAssert(num_cores > 0, "Failed to get number of cores from hwloc");
    return static_cast< unsigned >(num_cores);
}

auto Topology::getNHwThreads() const -> unsigned
{
    const auto num_threads = hwloc_get_nbobjs_by_type(m_topology.get(), HWLOC_OBJ_PU);
    throwingAssert(num_threads > 0, "Failed to get number of hardware threads from hwloc");
    return static_cast< unsigned >(num_threads);
}

auto Topology::getNodeMask(unsigned node) const -> Mask
{
    throwingAssert< std::out_of_range >(node < getNNodes(), "Node index out of bounds");
    auto node_mask = Mask{};
    node_mask.set(node);
    auto retval = Mask{};
    throwingAssert(not hwloc_cpuset_from_nodeset(m_topology.get(), retval, node_mask));
    return retval;
}

auto Topology::getCoreMask(unsigned core) const -> Mask
{
    throwingAssert< std::out_of_range >(core < getNCores(), "Core index out of bounds");
    const auto full_mask = getFullMask();
    return hwloc_get_obj_inside_cpuset_by_type(m_topology.get(), full_mask, HWLOC_OBJ_CORE, core)->cpuset;
}

void Topology::bindThread(const Mask& cpu_mask) const
{
    throwingAssert(not hwloc_set_cpubind(m_topology.get(), cpu_mask, HWLOC_CPUBIND_THREAD));
}

auto Topology::getThreadBinding() const -> Mask
{
    auto retval = Mask{};
    throwingAssert(not hwloc_get_cpubind(m_topology.get(), retval, HWLOC_CPUBIND_THREAD),
                   "Failed to get thread binding from hwloc");
    return retval;
}
} // namespace lstr::util::hwloc
#endif // L3STER_UTIL_NUMATOPOLOGY_HPP
