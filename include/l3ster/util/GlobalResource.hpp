#ifndef L3STER_UTIL_GLOBALRESOURCE_HPP
#define L3STER_UTIL_GLOBALRESOURCE_HPP

#include <concepts>
#include <optional>

namespace lstr::util
{
template < typename T >
class GlobalResource
{
public:
    template < typename... Args >
    static void initialize(Args&&... args)
        requires std::constructible_from< T, Args... >
    {
        getInstance().emplace(std::forward< Args >(args)...);
    }
    static bool isInitialized() { return getInstance().has_value(); }
    static T&   get() { return *getInstance(); }
    template < typename... Args >
    static T& getMaybeUninitialized(Args&&... args)
        requires std::constructible_from< T, Args... >
    {
        if (not isInitialized())
            initialize(std::forward< Args >(args)...);
        return get();
    }
    static void finalize() { getInstance().reset(); }

private:
    static std::optional< T >& getInstance()
    {
        static std::optional< T > instance;
        return instance;
    }
};
} // namespace lstr::util
#endif // L3STER_UTIL_GLOBALRESOURCE_HPP
