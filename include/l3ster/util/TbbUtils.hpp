#ifndef L3STER_UTIL_TBBUTILS_HPP
#define L3STER_UTIL_TBBUTILS_HPP

#include "l3ster/util/Concepts.hpp"
#include "l3ster/util/Ranges.hpp"

#include "oneapi/tbb.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>
#include <map>
#include <ranges>
#include <shared_mutex>

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

namespace detail
{
template < std::ranges::sized_range Range >
size_t largeGrainSizeHeuristic(Range&& range)
{
    const size_t par = oneapi::tbb::global_control::active_value(oneapi::tbb::global_control::max_allowed_parallelism);
    const auto   par_fp           = static_cast< double >(par);
    const size_t range_size       = std::ranges::size(range);
    const auto   range_size_fp    = static_cast< double >(range_size);
    const auto   range_bytes_fp   = range_size_fp * sizeof(std::ranges::range_value_t< Range >);
    const auto   bytes_per_worker = range_bytes_fp / par_fp;

    constexpr double shift                 = 1'250'000.;
    constexpr double scale                 = .0'000'038;
    constexpr double max_chunks_per_worker = 40.;
    const double     desired_chunks_per_worker =
        (max_chunks_per_worker - 1.) / (std::exp(-scale * (bytes_per_worker - shift)) + 1.) + 1.;
    const double grain_size = range_size_fp / par_fp / desired_chunks_per_worker;

    const size_t grain_lower_bnd = 1;
    const size_t grain_upper_bnd = range_size / par + 1;
    return std::clamp(static_cast< size_t >(std::llround(grain_size)), grain_lower_bnd, grain_upper_bnd);
}

template < std::input_iterator Iterator >
class IteratorCache
{
    template < std::ranges::sized_range Range >
    static size_t nSlotsHeuristic(Range&& range)
    {
        // Checkpoint every 4kB worth of elements
        constexpr size_t range_elem_size = sizeof(std::ranges::range_value_t< Range >);
        const size_t     range_size      = std::ranges::size(range);
        return std::max(size_t{1}, std::bit_ceil(range_size * range_elem_size >> 12));
    }

public:
    template < std::ranges::sized_range Range >
    explicit IteratorCache(Range&& range)
        : m_n_slots{nSlotsHeuristic(range)},
          m_range_size{std::ranges::size(range)},
          m_slot_size_log2{static_cast< size_t >(
              std::countr_zero(std::max(size_t{1}, std::bit_ceil(m_range_size + 1u) / m_n_slots)))},
          m_iter_ptrs{std::make_unique< std::atomic< Iterator* >[] >(m_n_slots)}
    {
        m_iter_ptrs[0].store(new Iterator{std::ranges::begin(range)}, std::memory_order_release);
    }
    IteratorCache(const IteratorCache&)                = delete;
    IteratorCache& operator=(const IteratorCache&)     = delete;
    IteratorCache(IteratorCache&&) noexcept            = default;
    IteratorCache& operator=(IteratorCache&&) noexcept = default;
    ~IteratorCache()
    {
        std::atomic_thread_fence(std::memory_order_acq_rel);
        for (size_t i = 0; i != m_n_slots; ++i)
            delete m_iter_ptrs[i].load(std::memory_order_relaxed);
    }

    Iterator getIter(size_t pos)
    {
        const auto slot_ind_desired = getSlotIndex(pos);
        auto       slot_ind         = slot_ind_desired;
        Iterator*  iter_ptr         = m_iter_ptrs[slot_ind].load(std::memory_order_acquire);
        while (not iter_ptr)
            iter_ptr = m_iter_ptrs[--slot_ind].load(std::memory_order_acquire);
        const size_t found_pos = slot_ind << m_slot_size_log2;
        const auto   diff      = pos - found_pos;
        const auto   retval    = std::next(*iter_ptr, diff);
        if (slot_ind != slot_ind_desired)
            putIter(pos, retval);
        return retval;
    }
    void putIter(size_t pos, Iterator iter)
    {
        const bool   is_boundary = static_cast< size_t >(std::countr_zero(pos)) >= m_slot_size_log2;
        const size_t slot_ind    = getSlotIndex(pos) + not is_boundary;
        const size_t dest_pos    = slot_ind << m_slot_size_log2;
        if (dest_pos >= m_range_size or m_iter_ptrs[slot_ind].load(std::memory_order_relaxed))
            return;
        auto      desired = new Iterator{std::next(iter, dest_pos - pos)};
        Iterator* expected{};
        if (not m_iter_ptrs[slot_ind].compare_exchange_strong(expected, desired, std::memory_order_acq_rel))
            delete desired;
    }

private:
    size_t getSlotIndex(size_t pos) const { return pos >> m_slot_size_log2; }

    size_t                                        m_n_slots, m_range_size, m_slot_size_log2;
    std::unique_ptr< std::atomic< Iterator* >[] > m_iter_ptrs;
};

template < std::ranges::sized_range Range >
IteratorCache(Range&& range) -> IteratorCache< std::ranges::iterator_t< Range > >;
} // namespace detail

template < std::ranges::sized_range Range, std::invocable< std::ranges::range_reference_t< Range > > Kernel >
void parallelFor(Range&& range, Kernel&& kernel)
{
    auto       iter_cache = detail::IteratorCache{range};
    const auto grain      = detail::largeGrainSizeHeuristic(range);
    const auto iter_space = detail::makeBlockedIterSpace(range, grain);

    oneapi::tbb::parallel_for(iter_space, [&](const auto& subrange) {
        auto       ind     = subrange.begin();
        auto       iter    = iter_cache.getIter(ind);
        const auto end_ind = subrange.end();
        for (; ind != end_ind; ++ind)
            std::invoke(kernel, *iter++);
        iter_cache.putIter(end_ind, iter);
    });
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
            return std::transform_reduce(std::execution::unseq,
                                         std::next(range.begin(), iter_range.begin()),
                                         std::next(range.begin(), iter_range.end()),
                                         value,
                                         reduction,
                                         transform);
        },
        reduction);
} // namespace lstr::util::tbb
} // namespace lstr::util::tbb
#endif // L3STER_UTIL_TBBUTILS_HPP
