#ifndef L3STER_MESH_CONSTANTS_HPP
#define L3STER_MESH_CONSTANTS_HPP

#include "l3ster/defs/Typedefs.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>

namespace lstr
{
#ifndef L3STER_ELEMENT_ORDERS
#define L3STER_ELEMENT_ORDERS 1, 2
#endif

// Define allowed element orders, must contain 1
constexpr inline auto element_orders = std::invoke([] {
    constexpr std::array declared_orders{L3STER_ELEMENT_ORDERS};
    static_assert(std::ranges::all_of(declared_orders, [](auto o) { return o > 0; }));
    constexpr bool   contains_one = std::ranges::find(declared_orders, 1) != end(declared_orders);
    constexpr size_t n_unique     = std::invoke([&] {
        auto declared_orders_copy = declared_orders;
        std::ranges::sort(declared_orders_copy);
        return std::distance(begin(declared_orders_copy), std::ranges::unique(declared_orders_copy).begin());
    });

    std::array< el_o_t, n_unique + not contains_one > retval{};
    auto                                              insert_it = begin(retval);
    if constexpr (not contains_one)
    {
        retval.front() = 1;
        ++insert_it;
    }
    auto declared_orders_copy = declared_orders;
    std::ranges::sort(declared_orders_copy);
    std::ranges::unique_copy(declared_orders_copy, insert_it);
    return retval;
});
} // namespace lstr
#endif // L3STER_MESH_CONSTANTS_HPP
