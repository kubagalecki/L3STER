#ifndef L3STER_UTIL_TBBUTILS_HPP
#define L3STER_UTIL_TBBUTILS_HPP

#include "l3ster/util/Concepts.hpp"
#include "l3ster/util/GlobalResource.hpp"
#include "l3ster/util/HwlocWrapper.hpp"
#include "l3ster/util/Ranges.hpp"
#include "l3ster/util/StringUtils.hpp"

#include "oneapi/tbb.h"

#include <cmath>
#include <iterator>
#include <ranges>

namespace lstr::util::tbb
{
namespace detail
{
template < std::ranges::sized_range Range >
using blocked_iter_space_t = oneapi::tbb::blocked_range< std::ranges::range_difference_t< Range > >;

template < std::ranges::sized_range Range >
auto makeBlockedIterSpace(Range&& range, size_t grain = 1) -> blocked_iter_space_t< Range >
{
    return {0, std::ranges::ssize(range), grain};
}
} // namespace detail

template < SizedRandomAccessRange_c Range, std::invocable< std::ranges::range_reference_t< Range > > Kernel >
void parallelFor(Range&& range, Kernel&& kernel)
{
    const auto iter_space = detail::makeBlockedIterSpace(range);
    using iter_space_t    = decltype(iter_space);
    oneapi::tbb::parallel_for(iter_space,
                              [&kernel, range_begin = std::ranges::begin(range)](const iter_space_t& subrange) {
                                  for (auto i : std::views::iota(subrange.begin(), subrange.end()))
                                      std::invoke(kernel, range_begin[i]);
                              });
}

template < std::ranges::sized_range Range, std::invocable< std::ranges::range_reference_t< Range > > Kernel >
void parallelFor(Range&& range, Kernel&& kernel)
{
    auto&& range_common = std::forward< Range >(range) | std::views::common;
    auto   begin = std::ranges::begin(range_common), end = std::ranges::end(range_common);
    oneapi::tbb::parallel_for_each(begin, end, std::forward< Kernel >(kernel));
}

template < SizedRandomAccessRange_c Range, std::random_access_iterator Iter, typename Kernel >
void parallelTransform(Range&& input_range, Iter output_iter, Kernel&& kernel)
    requires std::invocable< Kernel, range_const_reference_t< Range > > and
             std::output_iterator< Iter, std::invoke_result_t< Kernel, range_const_reference_t< Range > > >
{
    const auto iter_space  = detail::makeBlockedIterSpace(input_range);
    using iter_space_t     = decltype(iter_space);
    const auto range_begin = std::ranges::cbegin(input_range);
    oneapi::tbb::parallel_for(iter_space, [&](const iter_space_t& subrange) {
        for (auto i : std::views::iota(subrange.begin(), subrange.end()))
            output_iter[i] = std::invoke(kernel, range_begin[i]);
    });
}

template < typename Value, typename Zero, typename Reduction, typename Transform >
concept TransformReducible_c =
    std::is_invocable_r_v< Zero, Transform, Value > and std::is_invocable_r_v< Zero, Reduction, Zero, Zero >;

template < SizedRandomAccessRange_c Range,
           typename Zero,
           typename Reduction = std::plus<>,
           typename Transform = std::identity >
auto parallelTransformReduce(Range&& range, Zero identity, Reduction reduction = {}, Transform transform = {})
    -> std::decay_t< Zero >
    requires TransformReducible_c< range_const_reference_t< Range >, Zero, Reduction, Transform >
{
    const auto iter_space = detail::makeBlockedIterSpace(range);
    return oneapi::tbb::parallel_reduce(
        iter_space,
        identity,
        [&](const auto& iter_range, const auto& value) {
            return std::transform_reduce(std::next(range.begin(), iter_range.begin()),
                                         std::next(range.begin(), iter_range.end()),
                                         value,
                                         reduction,
                                         transform);
        },
        reduction);
}

enum struct Pin
{
    Node,
    Core,
    Thread,
    Unpinned
};

class PinningObserver : public oneapi::tbb::task_scheduler_observer
{
    class PinCache
    {
    public:
        PinCache() : my_mask_{GlobalResource< hwloc::Topology >::getMaybeUninitialized().getThreadBinding()} {}
        void bind(const hwloc::Mask& mask)
        {
            if (mask == my_mask_)
                return;
            GlobalResource< hwloc::Topology >::getMaybeUninitialized().bindThread(mask);
            my_mask_ = mask.copy();
        }

    private:
        hwloc::Mask my_mask_;
    };

public:
    PinningObserver(oneapi::tbb::task_arena& arena, Pin pin_type)
        : oneapi::tbb::task_scheduler_observer(arena),
          m_pin_type(pin_type),
          m_thread_masks(static_cast< size_t >(arena.max_concurrency()), cache_align)
    {
        if (pin_type == Pin::Unpinned)
            return;
        const auto thr_range = std::views::iota(0u, m_thread_masks.size());
        std::ranges::transform(thr_range, m_thread_masks.begin(), [&](unsigned i) { return getMask(i); });
        this->observe(true);
    }

    void on_scheduler_entry(bool is_worker) final
    {
        if (m_pin_type != Pin::Unpinned) [[likely]]
        {
            const auto  tid  = static_cast< unsigned >(oneapi::tbb::this_task_arena::current_thread_index());
            const auto& mask = m_thread_masks.at(tid);
            if (not is_worker)
            {
                const auto& topo = GlobalResource< hwloc::Topology >::getMaybeUninitialized();
                m_main_mask      = topo.getThreadBinding();
                topo.bindThread(mask);
            }
            else
            {
                thread_local PinCache pin_cache;
                pin_cache.bind(mask);
            }
        }
    }

    void on_scheduler_exit(bool is_worker) final
    {
        if (m_pin_type != Pin::Unpinned and not is_worker) [[likely]]
            GlobalResource< hwloc::Topology >::getMaybeUninitialized().bindThread(m_main_mask);
    }

private:
    // Round-robin assignment
    auto getMask(unsigned thread_index) const -> hwloc::Mask
    {
        const auto& topo   = GlobalResource< hwloc::Topology >::getMaybeUninitialized();
        auto        assign = [&topo, thread_index](unsigned (hwloc::Topology::*get_num_units)() const,
                                            hwloc::Mask (hwloc::Topology::*get_mask)(unsigned) const) mutable {
            const auto num_units = std::invoke(get_num_units, topo);
            while (thread_index >= num_units)
                thread_index -= num_units;
            return std::invoke(get_mask, topo, thread_index);
        };
        switch (m_pin_type)
        {
        case Pin::Node:
            return assign(&hwloc::Topology::getNNodes, &hwloc::Topology::getNodeMask);
        case Pin::Core:
            return assign(&hwloc::Topology::getNCores, &hwloc::Topology::getCoreMask);
        case Pin::Thread:
            return assign(&hwloc::Topology::getNHwThreads, &hwloc::Topology::getThreadMask);
        default:
            std::unreachable();
        }
    }

    Pin                       m_pin_type;
    hwloc::Mask               m_main_mask;
    ArrayOwner< hwloc::Mask > m_thread_masks;
};

inline auto getPinTypeFromEnv()
{
    using namespace std::string_view_literals;
    constexpr auto bind_strs      = std::array{"true"sv, "close"sv, "spread"sv};
    const auto     proc_bind      = getEnvOrEmpty("OMP_PROC_BIND");
    const auto     places         = getEnvOrEmpty("OMP_PLACES");
    constexpr auto starts_with_ci = [](std::string_view str, std::string_view prefix) {
        return equalCaseInsensitive(str.substr(0, prefix.size()), prefix);
    };
    if (std::ranges::any_of(bind_strs, [&](std::string_view bind_str) { return starts_with_ci(proc_bind, bind_str); }))
    {
        if (starts_with_ci(places, "sockets"))
            return Pin::Node;
        if (starts_with_ci(places, "cores"))
            return Pin::Core;
        return Pin::Thread;
    }
    return Pin::Unpinned;
}

inline auto getMaxParallelism()
{
    return oneapi::tbb::global_control::active_value(oneapi::tbb::global_control::max_allowed_parallelism);
}

inline auto getMaxParallelismCoresOnly() -> unsigned
{
    const auto        max_par_global = static_cast< unsigned >(getMaxParallelism());
    static const auto n_cores        = GlobalResource< hwloc::Topology >::getMaybeUninitialized().getNCores();
    return std::max(max_par_global, n_cores);
}
} // namespace lstr::util::tbb
#endif // L3STER_UTIL_TBBUTILS_HPP
