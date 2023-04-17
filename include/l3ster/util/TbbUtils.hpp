#ifndef L3STER_UTIL_TBBUTILS_HPP
#define L3STER_UTIL_TBBUTILS_HPP

#include "l3ster/util/Concepts.hpp"

#include "oneapi/tbb.h"

#include <iterator>
#include <ranges>

namespace lstr::util::tbb
{
namespace detail
{
template < SizedRandomAccessRange_c Range >
using blocked_iter_space_t = oneapi::tbb::blocked_range< std::ranges::range_difference_t< Range > >;

auto makeBlockedIterSpace(SizedRandomAccessRange_c auto&& range) -> blocked_iter_space_t< decltype(range) >
{
    return {0, std::ranges::ssize(range)};
}
} // namespace detail

void parallelFor(SizedRandomAccessRange_c auto&&                                            range,
                 std::invocable< std::ranges::range_reference_t< decltype(range) > > auto&& kernel)
{
    const auto iter_space = detail::makeBlockedIterSpace(range);
    using iter_space_t    = decltype(iter_space);
    oneapi::tbb::parallel_for(iter_space,
                              [&kernel, range_begin = std::ranges::begin(range)](const iter_space_t& subrange) {
                                  for (auto i : std::views::iota(subrange.begin(), subrange.end()))
                                      std::invoke(kernel, range_begin[i]);
                              });
}

void parallelTransform(SizedRandomAccessRange_c auto&& input_range, auto output_iter, auto&& kernel)
    requires std::random_access_iterator< decltype(output_iter) > and
             requires(decltype(*std::ranges::cbegin(input_range)) input_element) {
                 *output_iter = std::invoke(kernel, input_element);
             }
{
    const auto iter_space = detail::makeBlockedIterSpace(input_range);
    using iter_space_t    = decltype(iter_space);
    oneapi::tbb::parallel_for(iter_space,
                              [&, range_begin = std::ranges::cbegin(input_range)](const iter_space_t& subrange) {
                                  for (auto i : std::views::iota(subrange.begin(), subrange.end()))
                                      output_iter[i] = std::invoke(kernel, range_begin[i]);
                              });
} // namespace lstr::util::tbb

namespace detail
{
template < typename Iter, typename Reduced, typename Reduction, typename Transform >
class TransformReduceHelper
{
public:
    using range_diff_t = std::iterator_traits< Iter >::difference_type;

    TransformReduceHelper(Iter                           range_begin,
                          const Reduced&                 zero,
                          DecaysTo_c< Reduction > auto&& reduction,
                          DecaysTo_c< Transform > auto&& transform)
        : m_range_begin{range_begin},
          m_reduced_current{zero},
          m_zero_ptr{std::addressof(zero)},
          m_reduction{std::forward< decltype(reduction) >(reduction)},
          m_transform{std::forward< decltype(transform) >(transform)}
    {}
    TransformReduceHelper(const TransformReduceHelper& other, oneapi::tbb::split)
        : m_range_begin{other.m_range_begin},
          m_reduced_current{*other.m_zero_ptr},
          m_zero_ptr{other.m_zero_ptr},
          m_reduction{other.m_reduction},
          m_transform{other.m_transform}
    {}

    void join(const TransformReduceHelper& other)
    {
        m_reduced_current = std::invoke(m_reduction, m_reduced_current, other.m_reduced_current);
    }
    void operator()(const oneapi::tbb::blocked_range< range_diff_t >& subrange)
    {
        m_reduced_current = std::transform_reduce(std::execution::unseq,
                                                  std::next(m_range_begin, subrange.begin()),
                                                  std::next(m_range_begin, subrange.end()),
                                                  m_reduced_current,
                                                  m_reduction,
                                                  m_transform);
    }

    auto getReducedValue() const { return m_reduced_current; }

private:
    Iter           m_range_begin;
    Reduced        m_reduced_current;
    const Reduced* m_zero_ptr;
    Reduction      m_reduction;
    Transform      m_transform;
};
template < typename Iter, typename Reduced, typename Reduction, typename Transform >
TransformReduceHelper(Iter, const Reduced&, Reduction&&, Transform&&)
    -> TransformReduceHelper< Iter, std::decay_t< Reduced >, std::decay_t< Reduction >, std::decay_t< Transform > >;
} // namespace detail

auto parallelTransformReduce(SizedRandomAccessRange_c auto&&     range,
                             const std::copy_constructible auto& zero,
                             std::copy_constructible auto&&      reduction,
                             std::copy_constructible auto&&      transform = std::identity{})
    -> std::decay_t< decltype(zero) >
    requires requires(decltype(*std::ranges::cbegin(range)) range_element) {
        {
            std::invoke(transform, range_element)
        } -> std::convertible_to< decltype(zero) >;
        {
            std::invoke(reduction, zero, zero)
        } -> std::convertible_to< decltype(zero) >;
        {
            std::transform_reduce(std::ranges::cbegin(range), std::ranges::cend(range), zero, reduction, transform)
        } -> std::convertible_to< decltype(zero) >;
    }
{
    const auto iter_space = detail::makeBlockedIterSpace(range);
    auto       helper     = detail::TransformReduceHelper{std::ranges::cbegin(range),
                                                zero,
                                                std::forward< decltype(reduction) >(reduction),
                                                std::forward< decltype(transform) >(transform)};
    oneapi::tbb::parallel_reduce(iter_space, helper);
    return helper.getReducedValue();
}
} // namespace lstr::util::tbb
#endif // L3STER_UTIL_TBBUTILS_HPP
