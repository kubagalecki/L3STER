#ifndef L3STER_UTIL_INDEXMAP_HPP
#define L3STER_UTIL_INDEXMAP_HPP

#include <concepts>
#include <ranges>
#include <unordered_map>

namespace lstr
{
template < typename T, std::integral Index = std::ptrdiff_t >
class IndexMap
{
public:
    template < std::ranges::sized_range R >
    IndexMap(R&& value_range) : map{std::ranges::size(value_range)}
    {
        for (Index i{0}; const T& v : value_range)
            map.emplace(v, i++);
    }
    Index operator()(const T& entry) const noexcept { return map.find(entry)->second; }

private:
    std::unordered_map< T, Index > map;
};

template < std::ranges::sized_range R >
IndexMap(R&&) -> IndexMap< std::ranges::range_value_t< R > >;
} // namespace lstr
#endif // L3STER_UTIL_INDEXMAP_HPP
