#ifndef L3STER_UTIL_INDEXMAP_HPP
#define L3STER_UTIL_INDEXMAP_HPP

#include "l3ster/util/RobinHoodHashTables.hpp"

#include <concepts>
#include <ranges>
#include <unordered_map>

namespace lstr::util
{
template < typename T, std::integral Index = std::size_t >
class IndexMap
{
public:
    template < std::ranges::range R >
    IndexMap(R&& value_range)
    {
        if constexpr (std::ranges::sized_range< decltype(value_range) >)
            m_map.reserve(std::ranges::size(value_range));
        for (Index i = 0; const T& v : value_range)
            m_map.emplace(v, i++);
    }
    Index operator()(const T& entry) const noexcept { return m_map.at(entry); }

private:
    robin_hood::unordered_flat_map< T, Index > m_map;
};

template < std::ranges::sized_range R >
IndexMap(R&&) -> IndexMap< std::ranges::range_value_t< R > >;
} // namespace lstr::util
#endif // L3STER_UTIL_INDEXMAP_HPP
