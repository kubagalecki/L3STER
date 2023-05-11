#ifndef L3STER_UTIL_SETSTACKSIZE_HPP
#define L3STER_UTIL_SETSTACKSIZE_HPP

#include "l3ster/util/Assertion.hpp"

#include "oneapi/tbb.h"

#include "pthread.h"
#include "sys/resource.h"

namespace lstr::util
{
inline constexpr std::size_t default_stack_size = 1u << 23;

namespace detail
{
struct MaxStackSizeTracker
{
    static size_t get() { return access(); }
    static void   set(size_t value)
    {
        const auto prev_val = access();
        access()            = std::max(value, prev_val);
    }

private:
    static size_t& access()
    {
        static size_t value = default_stack_size;
        return value;
    };
};

template < size_t size >
struct MaxStackSizeRequest
{
    inline static const bool _ = std::invoke([] {
        MaxStackSizeTracker::set(size);
        return false;
    });
};
} // namespace detail

template < size_t size >
void requestStackSize()
{
    [[maybe_unused]] const auto request = detail::MaxStackSizeRequest< size >{};
}

namespace detail
{
inline size_t getCurrentThreadStackSize()
{
    pthread_attr_t attr;
    pthread_getattr_np(pthread_self(), &attr);
    size_t ss;
    pthread_attr_getstacksize(&attr, &ss);
    pthread_attr_destroy(&attr);
    return ss;
}

inline auto getStackSize()
{
    struct rlimit rl
    {};
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

    struct rlimit rl
    {};
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
        : m_tbb_global_control{
              oneapi::tbb::global_control{oneapi::tbb::global_control::thread_stack_size, std::bit_ceil(size)}}
    {
        detail::setMinStackSize(static_cast< rlim_t >(std::bit_ceil(size)));
    }

private:
    oneapi::tbb::global_control m_tbb_global_control;
};
} // namespace lstr::util
#endif // L3STER_UTIL_SETSTACKSIZE_HPP
