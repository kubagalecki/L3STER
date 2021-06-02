#ifndef L3STER_UTIL_COMMON_HPP
#define L3STER_UTIL_COMMON_HPP

#include <type_traits>

namespace lstr
{
template < typename... T >
struct OverloadSet : public T...
{
    using T::operator()...;
};
template < typename... T >
OverloadSet(T&&...) -> OverloadSet< T... >;

template < auto V >
requires(std::is_enum_v< decltype(V) >) struct EnumTag
{};
} // namespace lstr
#endif // L3STER_UTIL_COMMON_HPP
