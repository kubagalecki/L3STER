#ifndef L3STER_UTIL_WEAKCACHE_HPP
#define L3STER_UTIL_WEAKCACHE_HPP

#include "l3ster/util/Concepts.hpp"

#include <memory>
#include <unordered_map>

namespace lstr
{
template < typename Key, typename Value, typename Hash = std::hash< Key >, typename Equal = std::equal_to< Key > >
    requires Mapping_c< Hash, Key, size_t > and std::predicate< Equal, Key, Key >
class WeakCache
{
public:
    [[nodiscard]] auto contains(const Key& key) const -> bool { return m_cache.find(key) != m_cache.end(); }
    inline auto        get(const Key& key) const -> std::shared_ptr< Value >;
    auto               emplace(DecaysTo_c< Key > auto&& key, auto&&... args) -> std::shared_ptr< Value >
        requires std::constructible_from< Value, decltype(args)... >;

private:
    std::unordered_map< Key, std::weak_ptr< Value >, Hash, Equal > m_cache;
};

template < typename Key, typename Value, typename Hash, typename Equal >
auto WeakCache< Key, Value, Hash, Equal >::get(const Key& key) const -> std::shared_ptr< Value >
{
    const auto it = m_cache.find(key);
    return it != end(m_cache) ? it->second.lock() : std::shared_ptr< Value >{};
}

template < typename Key, typename Value, typename Hash, typename Equal >
auto WeakCache< Key, Value, Hash, Equal >::emplace(DecaysTo_c< Key > auto&& key, auto&&... args)
    -> std::shared_ptr< Value >
    requires std::constructible_from< Value, decltype(args)... >
{
    auto shared_ptr = std::make_shared< Value >(std::forward< decltype(args) >(args)...);
    m_cache.insert_or_assign(std::forward< decltype(key) >(key), shared_ptr);
    return shared_ptr;
}
} // namespace lstr
#endif // L3STER_UTIL_WEAKCACHE_HPP
