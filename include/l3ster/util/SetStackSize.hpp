#ifndef L3STER_UTIL_SETSTACKSIZE_HPP
#define L3STER_UTIL_SETSTACKSIZE_HPP

#include "oneapi/tbb.h"
#include "sys/resource.h"

#include <exception>
#include <utility>

namespace lstr::util
{
namespace detail
{
inline auto getStackSize()
{
    struct rlimit rl
    {};
    if (getrlimit(RLIMIT_STACK, &rl))
        throw std::runtime_error{"Could not determine the stack size"};
    return std::make_pair(rl.rlim_cur, rl.rlim_max);
}

inline void setMinStackSize(rlim_t requested_size)
{
    const auto [current_stack_size, max_stack_size] = getStackSize();
    if (requested_size < current_stack_size)
        return;
    if (requested_size > max_stack_size)
        throw std::runtime_error{"Requested stack size exceeds system limits"};

    struct rlimit rl
    {};
    rl.rlim_cur = requested_size;
    rl.rlim_max = max_stack_size;
    if (setrlimit(RLIMIT_STACK, &rl))
        throw std::runtime_error{"Could not increase the stack size to the desired size"};
}
} // namespace detail

class StackSizeGuard
{
public:
    StackSizeGuard(std::size_t size)
        : m_tbb_global_control{oneapi::tbb::global_control{oneapi::tbb::global_control::thread_stack_size, size}}
    {
        detail::setMinStackSize(static_cast< rlim_t >(size));
    }

private:
    oneapi::tbb::global_control m_tbb_global_control;
};
inline constexpr std::size_t default_stack_size = 1u << 23;
} // namespace lstr::util
#endif // L3STER_UTIL_SETSTACKSIZE_HPP
