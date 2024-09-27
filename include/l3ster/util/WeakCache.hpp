#ifndef L3STER_UTIL_WEAKCACHE_HPP
#define L3STER_UTIL_WEAKCACHE_HPP

#include "l3ster/util/Concepts.hpp"

#include <memory>
#include <unordered_map>

namespace lstr::util
{
template < typename Key, typename Value, typename Hash = std::hash< Key >, typename Equal = std::equal_to< Key > >
    requires Mapping_c< Hash, Key, size_t > and std::predicate< Equal, Key, Key >
class WeakCache
{
public:
    [[nodiscard]] auto contains(const Key& key) const -> bool { return m_cache.find(key) != m_cache.end(); }
    inline auto        get(const Key& key) const -> std::shared_ptr< Value >;
    template < DecaysTo_c< Key > K, typename... Args >
    auto emplace(K&& key, Args&&... args) -> std::shared_ptr< Value >
        requires std::constructible_from< Value, Args... >;

private:
    std::unordered_map< Key, std::weak_ptr< Value >, Hash, Equal > m_cache;
};

template < typename Key, typename Value, typename Hash, typename Equal >
    requires Mapping_c< Hash, Key, size_t > and std::predicate< Equal, Key, Key >
auto WeakCache< Key, Value, Hash, Equal >::get(const Key& key) const -> std::shared_ptr< Value >
{
    const auto it = m_cache.find(key);
    return it != end(m_cache) ? it->second.lock() : std::shared_ptr< Value >{};
}

template < typename Key, typename Value, typename Hash, typename Equal >
    requires Mapping_c< Hash, Key, size_t > and std::predicate< Equal, Key, Key >
             template < DecaysTo_c< Key > K, typename... Args >
             auto WeakCache< Key, Value, Hash, Equal >::emplace(K&& key, Args&&... args) -> std::shared_ptr< Value >
                 requires std::constructible_from< Value, Args... >
{
    auto shared_ptr = std::make_shared< Value >(std::forward< Args >(args)...);
    m_cache.insert_or_assign(std::forward< K >(key), shared_ptr);
    return shared_ptr;
}
} // namespace lstr::util
#endif // L3STER_UTIL_WEAKCACHE_HPP
