#ifndef L3STER_UTIL_TYPEID_HPP
#define L3STER_UTIL_TYPEID_HPP

namespace lstr
{
namespace detail
{
template < typename T >
struct TypeIDHelper
{
    static constexpr char tag{};
};
} // namespace detail
template < typename T >
inline constexpr const void* type_id_value = std::addressof(detail::TypeIDHelper< T >::tag);
} // namespace lstr
#endif // L3STER_UTIL_TYPEID_HPP
