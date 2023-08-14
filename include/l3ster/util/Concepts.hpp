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
// General
template < typename From, typename To >
concept DecaysTo_c = std::same_as< std::decay_t< From >, To >;
template < typename T >
concept Arithmetic_c = std::integral< T > or std::floating_point< T >;

// Designed to assert that a forwarding reference can be used to copy/move object `t`.
// Works both for:
// template <typename T> void foo(T&& t) requires ForwardConstructible_c<T> { auto t2 = std::forward<T>(t); }
// void foo(auto&& t) requires ForwardConstructible_c<decltype(t)> { auto t2 = std::forward< decltype(t) >(t); }
//
//  std::unique_ptr<...> up;
//  foo(up);             <- constraint failure, clear error message (hopefully)
//  foo(std::move(up));  <- OK
template < typename T >
concept ForwardConstructible_c = std::move_constructible< std::decay_t< T > > and
                                 (not std::is_lvalue_reference_v< T > or std::copy_constructible< std::decay_t< T > >);

// Range concepts
template < typename R, typename T >
concept RangeOfConvertibleTo_c = std::ranges::range< R > and std::convertible_to< std::ranges::range_value_t< R >, T >;
template < typename R, typename T >
concept SizedRangeOf_c = std::ranges::sized_range< R > and std::same_as< std::ranges::range_value_t< R >, T >;
template < typename R, typename T >
concept SizedRangeOfConvertibleTo_c = RangeOfConvertibleTo_c< R, T > and std::ranges::sized_range< R >;
template < typename R, typename T >
concept RandomAccessRangeOf =
    std::ranges::random_access_range< R > and std::same_as< std::ranges::range_value_t< R >, T >;
template < typename R, typename T >
concept ContiguousSizedRangeOf = std::ranges::contiguous_range< R > and
                                 std::same_as< std::ranges::range_value_t< R >, T > and std::ranges::sized_range< R >;
template < typename R >
concept IndexRange_c = SizedRangeOfConvertibleTo_c< R, std::size_t >;
template < typename R >
concept SizedRandomAccessRange_c = std::ranges::random_access_range< R > and std::ranges::sized_range< R >;

namespace detail
{
template < typename T >
inline constexpr bool is_array = false;
template < typename T, std::size_t N >
inline constexpr bool is_array< std::array< T, N > > = true;
template < typename T >
inline constexpr bool is_tuple = false;
template < typename... T >
inline constexpr bool is_tuple< std::tuple< T... > > = true;
template < typename T >
inline constexpr bool is_pair = false;
template < typename T1, typename T2 >
inline constexpr bool is_pair< std::pair< T1, T2 > > = true;
} // namespace detail

// Tuple-related cocnepts
template < typename T >
concept Array_c = detail::is_array< T >;
template < typename T, typename V >
concept ArrayOf_c = Array_c< T > and std::same_as< typename T::value_type, V >;
template < typename T >
concept Tuple_c = detail::is_tuple< T >;
template < typename T >
concept Pair_c = detail::is_pair< T >;

namespace detail
{
template < typename T, std::size_t I >
concept tuple_gettable = requires(T t) {
    typename std::tuple_element< I, T >;
    typename std::tuple_element_t< I, T >;
    {
        std::get< I >(t)
    } -> std::same_as< std::tuple_element_t< I, T > >;
};

template < typename, typename >
struct fold_tuple_gettable;

template < typename T, std::size_t... I >
    requires(tuple_gettable< T, I > && ...)
struct fold_tuple_gettable< T, std::index_sequence< I... > >
{};
} // namespace detail

template < typename T >
concept tuple_like = requires {
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
struct is_tuple_invocable< T, tuple_t > :
    std::conditional_t< std::invoke(
                            []< size_t... I >(std::index_sequence< I... >) {
                                return (std::is_invocable_v< T, std::tuple_element_t< I, tuple_t > > and ...);
                            },
                            std::make_index_sequence< std::tuple_size_v< tuple_t > >{}),
                        std::true_type,
                        std::false_type >
{};

template < typename R, typename T, typename tuple_t >
struct is_tuple_r_invocable : std::false_type
{};

template < typename R, typename T, tuple_like tuple_t >
struct is_tuple_r_invocable< R, T, tuple_t > :
    std::conditional_t< std::invoke(
                            []< size_t... I >(std::index_sequence< I... >) {
                                return (std::is_invocable_r_v< R, T, std::tuple_element_t< I, tuple_t > > and ...);
                            },
                            std::make_index_sequence< std::tuple_size_v< tuple_t > >{}),
                        std::true_type,
                        std::false_type >
{};
} // namespace detail

template < typename T, typename tuple_t >
concept tuple_invocable = detail::is_tuple_invocable< T, tuple_t >::value;

template < typename T, typename R, typename tuple_t >
concept tuple_r_invocable = detail::is_tuple_r_invocable< R, T, tuple_t >::value;

template < typename T, typename Domain, typename Range >
concept Mapping_c = requires(T f, Domain x) {
    {
        f(x)
    } -> std::convertible_to< Range >;
};

template < typename T, template < typename > typename Predicate >
concept predicate_trait_specialized = requires {
    {
        Predicate< std::decay_t< T > >::value
    } -> std::convertible_to< bool >;
};

// Execution policy concepts
template < typename T >
concept ExecutionPolicy_c = std::is_execution_policy_v< std::remove_cvref_t< T > >;
template < typename T >
concept SequencedPolicy_c = std::same_as< std::execution::sequenced_policy, std::remove_cvref_t< T > >;
template < typename T >
concept SimpleExecutionPolicy_c = std::same_as< std::execution::sequenced_policy, std::remove_cvref_t< T > > or
                                  std::same_as< std::execution::parallel_policy, std::remove_cvref_t< T > >;
} // namespace lstr
#endif // L3STER_UTIL_CONCEPTS_HPP
