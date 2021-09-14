#ifndef L3STER_UTIL_GLOBALRESOURCE_HPP
#define L3STER_UTIL_GLOBALRESOURCE_HPP

#include <concepts>
#include <optional>

namespace lstr
{
template < typename T >
class GlobalResource
{
public:
    template < typename... Args >
    requires std::constructible_from< T, Args... >
    static void init(Args&&... args) { getOpt().emplace(std::forward< Args >(args)...); }

    template < typename... Args >
    requires std::constructible_from< T, Args... >
    static T& getMaybeUninitialized(Args&&... args);
    static T& get() { return *getOpt(); }

    static bool isInitialized() { return getOpt().has_value(); }

private:
    static inline std::optional< T >& getOpt();
};

template < typename T >
template < typename... Args >
requires std::constructible_from< T, Args... > T& GlobalResource< T >::getMaybeUninitialized(Args&&... args)
{
    if (not isInitialized())
        init(std::forward< Args >(args)...);
    return get();
}

template < typename T >
std::optional< T >& GlobalResource< T >::getOpt()
{
    static std::optional< T > instance;
    return instance;
}
} // namespace lstr
#endif // L3STER_UTIL_GLOBALRESOURCE_HPP
