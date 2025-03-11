#ifndef L3STER_UTIL_SPATIALHASHTABLE_HPP
#define L3STER_UTIL_SPATIALHASHTABLE_HPP

#include "l3ster/common/Typedefs.h"
#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/Common.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace lstr::util
{
template < std::move_constructible T, size_t dim = 3 >
class SpatialHashTable
{
    using key_type                    = std::array< ptrdiff_t, 3 >;
    static constexpr val_t minimum_dx = std::numeric_limits< val_t >::epsilon() * (1 << sizeof(val_t));

public:
    using point_type = std::array< val_t, dim >;
    using value_type = std::pair< const point_type, T >;

    explicit SpatialHashTable(const point_type& dx, const point_type& origin = {}) : m_origin{origin}, m_dx{dx}
    {
        for (auto& d : m_dx)
            d = std::max(std::fabs(d), minimum_dx);
    }
    explicit SpatialHashTable(val_t dx = minimum_dx, const point_type& origin = {})
        : SpatialHashTable(makeFilledArray< dim >(dx), origin)
    {}

    template < typename... Args >
    void emplace(const point_type& point, Args&&... args)
        requires std::constructible_from< T, decltype(std::forward< decltype(args) >(args))... >
    {
        const auto key = snapToGrid(point);
        m_map[key].emplace_back(point, T(std::forward< Args >(args)...));
        ++m_size;
    }
    void insert(const point_type& point, const T& value) { emplace(point, value); }
    void insert(const point_type& point, T&& value) { emplace(point, std::move(value)); }

    auto proximate(const point_type& point) const
    {
        static constexpr auto offset1D = std::array{-1l, 0l, 1l};
        static constexpr auto offsets  = makeFilledArray< dim >(offset1D);
        const auto            key0     = snapToGrid(point);
        return std::apply(std::views::cartesian_product, offsets) |
               std::views::transform([this, key0](const auto& ofs) {
                   const auto ofs_array = std::apply([](auto... vals) { return std::array{vals...}; }, ofs);
                   auto       key       = key_type{};
                   std::ranges::copy(std::views::zip_transform(std::plus{}, key0, ofs_array), key.begin());
                   const auto iter = m_map.find(key);
                   return iter == m_map.end() ? std::span< const value_type >{} : std::span{iter->second};
               }) |
               std::views::join;
    }
    auto all() const
    {
        return m_map | std::views::transform([](const auto& p) { return std::span{p.second}; }) | std::views::join;
    }
    auto size() const -> size_t { return m_size; }

private:
    struct Hasher
    {
        static size_t operator()(const key_type& coords)
        {
            return robin_hood::hash_bytes(coords.data(), std::span{coords}.size_bytes());
        }
    };

    auto snapToGrid(const point_type& point) const -> key_type
    {
        constexpr auto snap = [](val_t x, val_t x0, val_t dx) {
            return std::lround((x - x0) / dx);
        };
        auto retval = key_type{};
        std::ranges::copy(std::views::zip_transform(snap, point, m_origin, m_dx), retval.begin());
        return retval;
    }

    robin_hood::unordered_flat_map< key_type, std::vector< value_type >, Hasher > m_map;
    std::array< val_t, dim >                                                      m_origin, m_dx;
    size_t                                                                        m_size = 0;
};
} // namespace lstr::util
#endif // L3STER_UTIL_SPATIALHASHTABLE_HPP
