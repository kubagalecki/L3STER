#ifndef L3STER_UTIL_SETSTACKSIZE_HPP
#define L3STER_UTIL_SETSTACKSIZE_HPP

#include "l3ster/util/Assertion.hpp"

#include "oneapi/tbb.h"
#include "sys/resource.h"

namespace lstr::util
{
namespace detail
{
inline auto getStackSize()
{
    struct rlimit rl{};
    const auto err_code = getrlimit(RLIMIT_STACK, &rl);
    util::throwingAssert(not err_code, "Could not determine the stack size");
    return std::make_pair(rl.rlim_cur, rl.rlim_max);
}

inline void setMinStackSize(rlim_t requested_size)
{
    const auto [current_stack_size, max_stack_size] = getStackSize();
    if (requested_size < current_stack_size)
        return;
    util::throwingAssert(requested_size <= max_stack_size, "Requested stack size exceeds system limits");

    struct rlimit rl{};
    rl.rlim_cur         = requested_size;
    rl.rlim_max         = max_stack_size;
    const auto err_code = setrlimit(RLIMIT_STACK, &rl);
    util::throwingAssert(not err_code, "Could not increase the stack size to the desired size");
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
