#ifndef L3STER_UTIL_WEAKCACHE_HPP
#define L3STER_UTIL_WEAKCACHE_HPP

#include "l3ster/util/Concepts.hpp"

#include <map>
#include <memory>

namespace lstr
{
template < typename Key, typename Value, typename Cmp = std::less< Key > >
class WeakCache
{
public:
    auto get(const Key& key) const -> std::shared_ptr< Value >;
    auto emplace(DecaysTo_c< Key > auto&& key, auto&&... args) -> std::shared_ptr< Value >
        requires std::constructible_from< Value, decltype(args)... >;
    auto push(DecaysTo_c< Key > auto&& key, DecaysTo_c< Value > auto&& value) -> std::shared_ptr< Value >;

private:
    std::map< Key, std::weak_ptr< Value >, Cmp > m_cache;
};

template < typename Key, typename Value, typename Cmp >
auto WeakCache< Key, Value, Cmp >::get(const Key& key) const -> std::shared_ptr< Value >
{
    if (const auto it = m_cache.find(key); it != end(m_cache))
        return it->second.lock();
    return {};
}

template < typename Key, typename Value, typename Cmp >
auto WeakCache< Key, Value, Cmp >::emplace(DecaysTo_c< Key > auto&& key, auto&&... args) -> std::shared_ptr< Value >
    requires std::constructible_from< Value, decltype(args)... >
{
    auto shared_ptr = std::make_shared< Value >(std::forward< decltype(args) >(args)...);
    m_cache.insert_or_assign(std::forward< decltype(key) >(key), shared_ptr);
    return shared_ptr;
}

template < typename Key, typename Value, typename Cmp >
auto WeakCache< Key, Value, Cmp >::push(DecaysTo_c< Key > auto&& key, DecaysTo_c< Value > auto&& value)
    -> std::shared_ptr< Value >
{
    return emplace(std::forward< decltype(key) >(key), std::forward< decltype(value) >(value));
}
} // namespace lstr
#endif // L3STER_UTIL_WEAKCACHE_HPP
