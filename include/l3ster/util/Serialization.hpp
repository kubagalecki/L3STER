#ifndef L3STER_UTIL_SERIALIZATION_HPP
#define L3STER_UTIL_SERIALIZATION_HPP

#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/Concepts.hpp"

#include <functional>
#include <string_view>
#include <type_traits>

namespace lstr::util
{
namespace detail
{
template < typename T >
inline constexpr bool is_serializable = std::invoke([] {
    if constexpr (TupleLike_c< T >)
    {
        constexpr auto helper = []< size_t... I >(std::index_sequence< I... >) {
            return (is_serializable< std::tuple_element_t< I, T > > and ...);
        };
        return helper(std::make_index_sequence< std::tuple_size_v< T > >{});
    }
    else if constexpr (std::ranges::range< T >)
        return is_serializable< std::ranges::range_value_t< T > >;
    else
        return std::is_trivially_copyable_v< T >;
});

template < typename T >
inline constexpr bool is_deserializable = std::invoke([] {
    if constexpr (TupleLike_c< T >)
    {
        constexpr auto helper = []< size_t... I >(std::index_sequence< I... >) {
            return (is_deserializable< std::tuple_element_t< I, T > > and ...);
        };
        return helper(std::make_index_sequence< std::tuple_size_v< T > >{});
    }
    else if constexpr (std::ranges::range< T >)
        return std::constructible_from< std::ranges::iterator_t< T >, std::ranges::iterator_t< T > > and
               is_deserializable< std::ranges::range_value_t< T > >;
    else
        return std::is_default_constructible_v< T > and std::is_trivially_copyable_v< T >;
});
} // namespace detail

template < typename T >
concept Serializable_c = detail::is_serializable< T >;
template < typename T >
concept Deserializable_c = detail::is_deserializable< T >;

template < typename T, std::output_iterator< char > It >
auto serialize(T&& data, It out_iter) -> It
    requires Serializable_c< std::remove_cvref_t< T > >
{
    using value_type = std::remove_cvref_t< T >;
    if constexpr (TupleLike_c< value_type >)
    {
        const auto serialize_by_index = [&]< size_t I >(std::integral_constant< size_t, I >) {
            out_iter = serialize(std::get< I >(data), out_iter);
        };
        const auto fold_indices = [&serialize_by_index]< size_t... I >(std::index_sequence< I... >) {
            (serialize_by_index(std::integral_constant< size_t, I >{}), ...);
        };
        fold_indices(std::make_index_sequence< std::tuple_size_v< value_type > >{});
    }
    else if constexpr (std::ranges::range< T >)
    {
        const auto size = static_cast< size_t >(std::ranges::distance(data));
        out_iter        = serialize(size, out_iter);
        for (auto&& value : std::forward< T >(data))
            out_iter = serialize(std::forward< decltype(value) >(value), out_iter);
    }
    else if constexpr (std::is_trivially_copyable_v< value_type >)
        out_iter = std::copy_n(reinterpret_cast< const char* >(&data), sizeof(T), out_iter);
    else
        static_assert(util::always_false< std::tuple_size_v< std::tuple< T > > >);
    return out_iter;
}

namespace detail
{
template < typename T >
auto deserializeImpl(std::string_view& serial_data) -> T
    requires Deserializable_c< std::remove_cvref_t< T > >
{
    using value_type = std::remove_cvref_t< T >;
    if constexpr (TupleLike_c< value_type >)
    {
        const auto deserialize_by_index = [&]< size_t I >(std::integral_constant< size_t, I >) {
            using type = std::tuple_element_t< I, value_type >;
            return deserializeImpl< type >(serial_data);
        };
        const auto fold_indices = [&deserialize_by_index]< size_t... I >(std::index_sequence< I... >) {
            return value_type{deserialize_by_index(std::integral_constant< size_t, I >{})...};
        };
        return fold_indices(std::make_index_sequence< std::tuple_size_v< value_type > >{});
    }
    else if constexpr (std::ranges::range< value_type >)
    {
        using range_value_type = std::ranges::range_value_t< value_type >;
        constexpr auto szsz    = sizeof(size_t);
        throwingAssert(serial_data.size() >= szsz);
        size_t size{};
        std::memcpy(&size, serial_data.data(), szsz);
        serial_data.remove_prefix(szsz);
        auto ctor_range =
            std::views::iota(0uz, size) |
            std::views::transform([&](size_t) { return deserializeImpl< range_value_type >(serial_data); }) |
            std::views::common;
        return value_type{ctor_range.begin(), ctor_range.end()};
    }
    else if constexpr (std::is_trivially_copyable_v< value_type >)
    {
        throwingAssert(serial_data.size() >= sizeof(value_type));
        value_type retval;
        std::memcpy(&retval, serial_data.data(), sizeof(value_type));
        serial_data.remove_prefix(sizeof(value_type));
        return retval;
    }
    else
        static_assert(util::always_false< std::tuple_size_v< std::tuple< T > > >);
}
} // namespace detail

template < typename T >
auto deserialize(std::string_view serial_data) -> T
    requires Deserializable_c< std::remove_cvref_t< T > >
{
    return detail::deserializeImpl< T >(serial_data);
}
} // namespace lstr::util
#endif // L3STER_UTIL_SERIALIZATION_HPP
