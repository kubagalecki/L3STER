#ifndef L3STER_UTIL_CONCEPTS_HPP
#define L3STER_UTIL_CONCEPTS_HPP

#include <array>
#include <concepts>
#include <tuple>
#include <utility>

namespace lstr
{
namespace detail
{
template < typename T >
struct is_array : std::false_type
{};
template < typename T, std::size_t N >
struct is_array< std::array< T, N > > : std::true_type
{};
} // namespace detail

template < typename T >
concept array = detail::is_array< T >::value;

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail
{
    template < typename T >
    struct is_tuple : std::false_type
    {};
    template < typename... T >
    struct is_tuple< std::tuple< T... > > : std::true_type
    {};
} // namespace detail

template < typename T >
concept tuple = detail::is_tuple< T >::value;

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail
{
    template < typename T >
    struct is_pair : std::false_type
    {};
    template < typename T1, typename T2 >
    struct is_pair< std::pair< T1, T2 > > : std::true_type
    {};
} // namespace detail

template < typename T >
concept pair = detail::is_pair< T >::value;

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail
{
    template < typename T, std::size_t I >
    concept tuple_gettable = requires(T t)
    {
        typename std::tuple_element< I, T >;
        typename std::tuple_element_t< I, T >;
        {
            std::get< I >(t)
            } -> std::same_as< std::tuple_element_t< I, T > >;
    };

    template < typename, typename >
    struct fold_tuple_gettable;

    template < typename T, std::size_t... I >
    requires(tuple_gettable< T, I > && ...) struct fold_tuple_gettable< T, std::index_sequence< I... > >
    {};
} // namespace detail

template < typename T >
concept tuple_like = requires
{
    std::tuple_size< T >::value;
    std::tuple_size_v< T >;
    {
        std::tuple_size_v< T >
        } -> std::convertible_to< std::size_t >;
    typename detail::fold_tuple_gettable< T, std::make_index_sequence< std::tuple_size_v< T > > >;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template < typename T, typename Domain, typename Range >
concept mapping = requires(T f, Domain x)
{
    {
        f(x)
        } -> std::convertible_to< Range >;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template < typename T, template < typename > typename Predicate >
concept predicate_trait_specialized = requires
{
    {
        Predicate< T >::value
        } -> std::convertible_to< bool >;
};

} // namespace lstr

#endif // L3STER_UTIL_CONCEPTS_HPP
