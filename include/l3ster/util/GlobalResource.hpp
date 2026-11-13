#ifndef L3STER_UTIL_GLOBALRESOURCE_HPP
#define L3STER_UTIL_GLOBALRESOURCE_HPP

#include <concepts>

namespace lstr::util
{
template < std::default_initializable T >
class GlobalResource
{
public:
    static T& get()
    {
        static T instance;
        return instance;
    }
};
} // namespace lstr::util
#endif // L3STER_UTIL_GLOBALRESOURCE_HPP
