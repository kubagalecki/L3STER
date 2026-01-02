#ifndef L3STER_UTIL_SERIALIZATION_HPP
#define L3STER_UTIL_SERIALIZATION_HPP

#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/Concepts.hpp"

#include <string_view>
#include <type_traits>

namespace lstr::util
{
template < typename T >
struct Serializer;

template < typename T >
concept Serializable_c = requires(Serializer< T > serializer, const T data, char* out_iter) {
    { serializer(data, out_iter) } -> std::same_as< char* >;
};

template < typename T >
    requires std::is_trivially_copyable_v< T > and (not TupleLike_c< T >) and (not std::ranges::range< T >)
struct Serializer< T >
{
    template < std::output_iterator< char > Iter >
    static constexpr auto operator()(const T& data, Iter out_iter) -> Iter
    {
        return std::copy_n(reinterpret_cast< const char* >(&data), sizeof(T), out_iter);
    }
};

template < std::ranges::range T >
    requires(not TupleLike_c< T >) and Serializable_c< std::ranges::range_value_t< T > >
struct Serializer< T >
{
    template < typename R, std::output_iterator< char > Iter >
    static constexpr auto operator()(R&& data, Iter out_iter) -> Iter
        requires std::same_as< T, std::remove_cvref_t< R > >
    {
        using value_type = std::ranges::range_value_t< T >;
        const auto size  = static_cast< size_t >(std::ranges::distance(data));
        out_iter         = Serializer< size_t >{}(size, out_iter);
        for (const auto& value : data)
            out_iter = Serializer< value_type >{}(value, out_iter);
        return out_iter;
    }
};

template < TupleLike_c T >
struct Serializer< T >
{
    template < std::output_iterator< char > Iter >
    static constexpr auto operator()(const T& data, Iter out_iter) -> Iter
    {
        const auto serialize_by_index = [&]< size_t I >(std::integral_constant< size_t, I >) {
            auto serializer = Serializer< std::remove_cvref_t< std::tuple_element_t< I, T > > >{};
            out_iter        = serializer(std::get< I >(data), out_iter);
        };
        const auto fold_indices = [&serialize_by_index]< size_t... I >(std::index_sequence< I... >) {
            (serialize_by_index(std::integral_constant< size_t, I >{}), ...);
        };
        fold_indices(std::make_index_sequence< std::tuple_size_v< T > >{});
        return out_iter;
    }
};

template < typename T >
struct Deserializer;

template < typename T >
concept Deserializable_c = requires(Deserializer< T > deserializer, std::string_view& serial_data) {
    { deserializer(serial_data) } -> std::same_as< T >;
};

template < typename T >
    requires std::is_trivially_copyable_v< T > and std::is_default_constructible_v< T > and (not TupleLike_c< T >) and
             (not std::ranges::range< T >)
struct Deserializer< T >
{
    static constexpr auto operator()(std::string_view& serial_data) -> T
    {
        throwingAssert(serial_data.size() >= sizeof(T));
        T retval;
        std::memcpy(&retval, serial_data.data(), sizeof(T));
        serial_data.remove_prefix(sizeof(T));
        return retval;
    }
};

template < std::ranges::range T >
    requires(not TupleLike_c< T >) and Deserializable_c< std::ranges::range_value_t< T > >
struct Deserializer< T >
{
    static constexpr auto operator()(std::string_view& serial_data) -> T
    {
        using range_value_t = std::ranges::range_value_t< T >;
        constexpr auto szsz = sizeof(size_t);
        throwingAssert(serial_data.size() >= szsz);
        size_t size{};
        std::memcpy(&size, serial_data.data(), szsz);
        serial_data.remove_prefix(szsz);
        auto ctor_range = std::views::iota(0uz, size) |
                          std::views::transform([&](size_t) { return Deserializer< range_value_t >{}(serial_data); }) |
                          std::views::common;
        return T{ctor_range.begin(), ctor_range.end()};
    }
};

template < TupleLike_c T >
struct Deserializer< T >
{
    static constexpr auto operator()(std::string_view& serial_data) -> T
    {
        const auto deserialize_by_index = [&]< size_t I >(std::integral_constant< size_t, I >) {
            using type = std::tuple_element_t< I, T >;
            return Deserializer< type >{}(serial_data);
        };
        const auto fold_indices = [&deserialize_by_index]< size_t... I >(std::index_sequence< I... >) {
            return T{deserialize_by_index(std::integral_constant< size_t, I >{})...};
        };
        return fold_indices(std::make_index_sequence< std::tuple_size_v< T > >{});
    }
};

template < Serializable_c T, std::output_iterator< char > It >
auto serialize(const T& data, It out_iter) -> It
{
    return Serializer< T >{}(data, out_iter);
}

template < Deserializable_c T >
auto deserialize(std::string_view serial_data) -> T
{
    return Deserializer< T >{}(serial_data);
}
} // namespace lstr::util
#endif // L3STER_UTIL_SERIALIZATION_HPP
