#ifndef L3STER_UTIL_SETSTACKSIZE_HPP
#define L3STER_UTIL_SETSTACKSIZE_HPP

#include "sys/resource.h"

#include <exception>
#include <utility>

namespace lstr
{
inline auto getStackSize()
{
    struct rlimit rl{};
    if (getrlimit(RLIMIT_STACK, &rl))
        throw std::runtime_error{"Could not determine the stack size"};
    return std::make_pair(rl.rlim_cur, rl.rlim_max);
}

inline void setMinStackSize(rlim_t requested_size)
{
    const auto& [current_stack_size, max_stack_size] = getStackSize();
    if (requested_size < current_stack_size)
        return;
    if (requested_size > max_stack_size)
        throw std::runtime_error{"Requested stack size exceeds system limits"};

    struct rlimit rl{};
    rl.rlim_cur = requested_size;
    rl.rlim_max = max_stack_size;
    if (setrlimit(RLIMIT_STACK, &rl))
        throw std::runtime_error{"Could not increase the stack size to the desired size"};
}
} // namespace lstr
#endif // L3STER_UTIL_SETSTACKSIZE_HPP
