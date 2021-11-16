#ifndef L3STER_UTIL_CONCEPTS_HPP
#define L3STER_UTIL_CONCEPTS_HPP

#include <array>
#include <concepts>
#include <execution>
#include <ranges>
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

template < typename T, typename V >
concept array_of = array< T > and std::same_as< typename T::value_type, V >;

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

namespace detail
{
    template < typename T, typename tuple_t >
    struct is_tuple_invocable : std::false_type
    {};

    template < typename T, tuple_like tuple_t >
        struct is_tuple_invocable< T, tuple_t > : std::conditional_t < []< size_t... I >(std::index_sequence< I... >)
    {
        return (std::is_invocable_v< T, std::tuple_element_t< I, tuple_t > > and ...);
    }(std::make_index_sequence< std::tuple_size_v< tuple_t > >{}), std::true_type, std::false_type > {};

    template < typename R, typename T, typename tuple_t >
    struct is_tuple_r_invocable : std::false_type
    {};

    template < typename R, typename T, tuple_like tuple_t >
        struct is_tuple_r_invocable< R, T, tuple_t > :
        std::conditional_t < []< size_t... I >(std::index_sequence< I... >)
    {
        return (std::is_invocable_r_v< R, T, std::tuple_element_t< I, tuple_t > > and ...);
    }(std::make_index_sequence< std::tuple_size_v< tuple_t > >{}), std::true_type, std::false_type > {};
} // namespace detail

template < typename T, typename tuple_t >
concept tuple_invocable = detail::is_tuple_invocable< T, tuple_t >::value;

template < typename T, typename R, typename tuple_t >
concept tuple_r_invocable = detail::is_tuple_r_invocable< R, T, tuple_t >::value;

template < typename T, typename Domain, typename Range >
concept mapping = requires(T f, Domain x)
{
    {
        f(x)
        } -> std::convertible_to< Range >;
};

template < typename T, template < typename > typename Predicate >
concept predicate_trait_specialized = requires
{
    {
        Predicate< std::decay_t< T > >::value
        } -> std::convertible_to< bool >;
};

template < typename R, typename V >
concept random_access_typed_range =
    std::ranges::random_access_range< R > && std::same_as< std::ranges::range_value_t< R >, V >;

template < typename T >
concept arithmetic = std::is_arithmetic_v< T >;

template < typename T >
concept ExecutionPolicy_c = std::is_execution_policy_v< T >;
} // namespace lstr
#endif // L3STER_UTIL_CONCEPTS_HPP
