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
    static constexpr auto bitmap_deleter = [](hwloc_bitmap_t ptr) {
        hwloc_bitmap_free(ptr);
    };
    using bitmap_t  = std::remove_pointer_t< hwloc_bitmap_t >;
    using deleter_t = std::remove_const_t< decltype(bitmap_deleter) >;

    struct UnallocatedTag
    {};
    struct CopyTag
    {};
    static constexpr auto unallocated = UnallocatedTag{};

    Mask(UnallocatedTag) {}

public:
    static constexpr auto make_copy = CopyTag{};

    Mask() : m_bitmap{hwloc_bitmap_alloc()} { assertNotNull(); }
    Mask(hwloc_const_bitmap_t bitmap, CopyTag) : m_bitmap(hwloc_bitmap_dup(bitmap)) { assertNotNull(); }

    auto getRaw() const -> hwloc_const_bitmap_t { return m_bitmap.get(); }
    auto getRaw() -> hwloc_bitmap_t { return m_bitmap.get(); }

    void set(unsigned index) { throwingAssert(not hwloc_bitmap_set(getRaw(), index)); }
    void unset(unsigned index) { throwingAssert(not hwloc_bitmap_clr(getRaw(), index)); }
    auto test(unsigned index) const -> bool { return hwloc_bitmap_isset(getRaw(), index); }
    auto weight() const -> unsigned { return static_cast< unsigned >(hwloc_bitmap_weight(m_bitmap.get())); }
    void clear() { hwloc_bitmap_zero(m_bitmap.get()); }

    inline auto operator!() const -> Mask;
    inline auto copy() const -> Mask;

private:
    void assertNotNull(std::source_location sl = std::source_location::current()) const
    {
        throwingAssert(bool{m_bitmap}, "Hwloc bitmap allocation failed", sl);
    }

    std::unique_ptr< bitmap_t, deleter_t > m_bitmap;
};

inline bool operator==(const Mask& b1, const Mask& b2)
{
    return hwloc_bitmap_isequal(b1.getRaw(), b2.getRaw());
}

inline auto operator<=>(const Mask& b1, const Mask& b2) -> std::strong_ordering
{
    const auto cmp = hwloc_bitmap_compare(b1.getRaw(), b2.getRaw());
    if (cmp == -1)
        return std::strong_ordering::less;
    if (cmp == 0)
        return std::strong_ordering::equal;
    if (cmp == 1)
        return std::strong_ordering::greater;
    std::unreachable();
}

#define L3STER_DEFINE_HWLOC_BMP_BINOP(symbol, op)                                                                      \
    inline Mask operator symbol(const Mask& b1, const Mask& b2)                                                        \
    {                                                                                                                  \
        auto retval = Mask{};                                                                                          \
        op(retval.getRaw(), b1.getRaw(), b2.getRaw());                                                                 \
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

    auto getFullMask() const -> Mask { return {hwloc_topology_get_topology_cpuset(m_topology.get()), Mask::make_copy}; }
    inline auto getNodeMask(unsigned node) const -> Mask;
    inline auto getCoreMask(unsigned core) const -> Mask;
    inline auto getThreadMask(unsigned thread) const -> Mask;

    inline void bindThread(const Mask& cpu_mask) const;
    inline auto getThreadBinding() const -> Mask;
    inline auto getLastThreadLocation() const -> unsigned;

private:
    using topology_t = std::remove_pointer_t< hwloc_topology_t >;

    std::unique_ptr< topology_t, decltype(&hwloc_topology_destroy) > m_topology;
};

auto Mask::operator!() const -> Mask
{
    auto retval = Mask{};
    hwloc_bitmap_not(retval.getRaw(), getRaw());
    return retval;
}

auto Mask::copy() const -> Mask
{
    auto retval = Mask{unallocated};
    retval.m_bitmap.reset(hwloc_bitmap_dup(getRaw()));
    return retval;
}

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
    throwingAssert(not hwloc_cpuset_from_nodeset(m_topology.get(), retval.getRaw(), node_mask.getRaw()));
    return retval;
}

auto Topology::getCoreMask(unsigned core) const -> Mask
{
    throwingAssert< std::out_of_range >(core < getNCores(), "Core index out of bounds");
    const auto full_mask = getFullMask();
    return {hwloc_get_obj_inside_cpuset_by_type(m_topology.get(), full_mask.getRaw(), HWLOC_OBJ_CORE, core)->cpuset,
            Mask::make_copy};
}

inline auto Topology::getThreadMask(unsigned thread) const -> Mask
{
    // Manually setting the bit works better than hwloc_get_obj_inside_cpuset_by_type for round-robin thread assignment,
    // since consecutive threads end up on different cores (standard linux logical-to-hardware thread mapping)
    throwingAssert< std::out_of_range >(thread < getNHwThreads(), "Thread index out of bounds");
    auto retval = Mask{};
    retval.set(thread);
    return retval;
}

void Topology::bindThread(const Mask& cpu_mask) const
{
    throwingAssert(not hwloc_set_cpubind(m_topology.get(), cpu_mask.getRaw(), HWLOC_CPUBIND_THREAD));
}

auto Topology::getThreadBinding() const -> Mask
{
    auto retval = Mask{};
    throwingAssert(not hwloc_get_cpubind(m_topology.get(), retval.getRaw(), HWLOC_CPUBIND_THREAD),
                   "Failed to get thread binding from hwloc");
    return retval;
}

auto Topology::getLastThreadLocation() const -> unsigned
{
    auto mask = Mask{};
    hwloc_get_last_cpu_location(m_topology.get(), mask.getRaw(), HWLOC_CPUBIND_THREAD);
    throwingAssert(mask.weight() == 1);
    const auto index_range = std::views::iota(0u, getNHwThreads());
    return *std::ranges::find_if(index_range, [&](unsigned i) { return mask.test(i); });
}
} // namespace lstr::util::hwloc
#endif // L3STER_UTIL_NUMATOPOLOGY_HPP
