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
    static T& get() { return getInstance().value(); }
    template < typename... Args >
    static T& getMaybeUninitialized(Args&&... args)
        requires std::constructible_from< T, Args... >
    {
        auto& instance = getInstance();
        if (not instance)
            instance.emplace(std::forward< Args >(args)...);
        return *instance;
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
