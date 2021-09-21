// Meta-programming helper classes

#ifndef L3STER_UTIL_META_HPP
#define L3STER_UTIL_META_HPP

#include "Common.hpp"
#include "Concepts.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <tuple>
#include <utility>
#include <variant>

namespace lstr
{
template < typename... T >
struct type_set
{};

template < auto... V >
struct value_set
{};

namespace detail
{
template < std::integral T, T first, T last >
requires(first <= last) constexpr auto make_interval_array()
{
    std::array< T, last - first + 1 > interval;
    std::iota(interval.begin(), interval.end(), first);
    return interval;
}

template < array auto A >
requires std::integral< typename decltype(A)::value_type >
constexpr auto int_seq_from_array()
{
    return []< size_t... I >(std::index_sequence< I... >)
    {
        return std::integer_sequence< typename decltype(A)::value_type, A[I]... >{};
    }
    (std::make_index_sequence< A.size() >{});
}
} // namespace detail

template < std::integral T, T first, T last >
requires(first <= last) using int_seq_interval =
    decltype(detail::int_seq_from_array< detail::make_interval_array< T, first, last >() >());

namespace detail
{
template < typename T >
struct Constify
{
    const T operator()(const T& in) requires(not std::is_pointer_v< T >) { return in; }
    const std::pointer_traits< T >::element_type* operator()(T in) requires(std::is_pointer_v< T >) { return in; }
    using type = decltype(std::declval< Constify< T > >()(std::declval< T >()));
};
} // namespace detail

// assumes types in pack T are unique; TODO: write concept which checks this assumption
template < typename... T >
constexpr auto constifyVariant(const std::variant< T... >& v)
{
    using const_variant_t = std::variant< typename detail::Constify< T >::type... >;
    return std::visit< const_variant_t >(OverloadSet{detail::Constify< T >{}...}, v);
}

// Functionality related to parametrizing over all combinations of a pack of nttp arrays
template < array auto A >
requires std::totally_ordered< typename decltype(A)::value_type >
struct unique_els
{
private:
    using A_t       = decltype(A);
    using A_value_t = A_t::value_type;

    static constexpr A_t A_sorted = [] {
        auto A_cp = A;
        std::sort(A_cp.begin(), A_cp.end());
        return A_cp;
    }();

public:
    static constexpr size_t size = [] {
        auto A_temp = A_sorted;
        return std::distance(A_temp.begin(), std::unique(A_temp.begin(), A_temp.end()));
    }();

    static constexpr std::array< A_value_t, size > value = [] {
        std::array< A_value_t, size > ret;
        std::unique_copy(A_sorted.cbegin(), A_sorted.cend(), ret.begin());
        return ret;
    }();
};

template < array auto A >
constexpr inline auto unique_els_v = unique_els< A >::value;

template < array auto... A >
requires((A.size() == std::get< 0 >(std::tie(A...)).size()) and ...) struct tuplify
{
    static constexpr size_t N = (A.size(), ...);
    using tuple_t             = std::tuple< typename decltype(A)::value_type... >;
    using value_type          = std::array< tuple_t, N >;

private:
    template < size_t I >
    struct tuplify_index
    {
        static constexpr tuple_t value{std::make_tuple(std::get< I >(A)...)};
    };

    template < size_t... I >
    static consteval auto tuplify_indices(std::index_sequence< I... >)
    {
        return value_type{tuplify_index< I >::value...};
    }

public:
    static constexpr auto value = tuplify_indices(std::make_index_sequence< N >{});
};

template < array auto... A >
requires(std::totally_ordered< typename decltype(A)::value_type >and...) and
    ((A.size() > 0) and ...) struct all_combinations
{
    static constexpr auto unique_A = std::make_tuple(unique_els_v< A >...);
    static constexpr auto size     = (unique_els< A >::size * ...);

private:
    static constexpr auto unique_A_sizes = std::array{unique_els< A >::size...};
    static constexpr auto reps           = [] {
        std::array< size_t, sizeof...(A) > ret;
        ret.back() = 1;
        std::partial_sum(unique_A_sizes.crbegin(), unique_A_sizes.crend() - 1, ret.rbegin() + 1, std::multiplies<>{});
        return ret;
    }();

    template < size_t I >
    static consteval auto extend_array()
    {
        std::array< typename std::tuple_element_t< I, decltype(unique_A) >::value_type, size > ret;
        std::generate(ret.begin(), ret.end(), [rep_ind = 0u, arr_ind = 0u]() mutable {
            constexpr auto& current_array = std::get< I >(unique_A);
            auto            ret_v         = current_array[arr_ind];
            ++rep_ind;
            if (rep_ind == reps[I])
            {
                rep_ind = 0;
                ++arr_ind;
                if (arr_ind == current_array.size())
                    arr_ind = 0;
            }
            return ret_v;
        });
        return ret;
    }

    template < size_t... I >
    static consteval auto compute(std::index_sequence< I... >)
    {
        return tuplify< extend_array< I >()... >::value;
    }

public:
    static constexpr auto value = compute(std::make_index_sequence< sizeof...(A) >{});
};

template < template < typename... > typename T, tuple Params >
struct apply_types
{
private:
    template < typename >
    struct deduction_helper;
    template < size_t... I >
    struct deduction_helper< std::index_sequence< I... > >
    {
        using type = T< std::tuple_element_t< I, Params >... >;
    };

public:
    using type = deduction_helper< std::make_index_sequence< std::tuple_size_v< Params > > >::type;
};

template < template < typename... > typename T, tuple Params >
using apply_types_t = apply_types< T, Params >::type;

template < template < auto... > typename Inner, template < typename... > typename Outer, tuple_like auto... Params >
struct parametrize_over_combinations
{
private:
    static constexpr auto combinations     = all_combinations< Params... >::value;
    static constexpr auto combination_size = all_combinations< Params... >::size;

    template < size_t I >
    struct apply_inner
    {
        template < typename >
        struct deduction_helper;
        template < size_t... Idx >
        struct deduction_helper< std::index_sequence< Idx... > >
        {
            using type = Inner< std::get< Idx >(combinations[I])... >;
        };

        using type = deduction_helper< std::make_index_sequence< sizeof...(Params) > >::type;
    };

    template < typename >
    struct deduction_helper;
    template < size_t... I >
    struct deduction_helper< std::index_sequence< I... > >
    {
        using type = apply_types_t< Outer, std::tuple< typename apply_inner< I >::type... > >;
    };

public:
    using type = deduction_helper< std::make_index_sequence< combination_size > >::type;
};

template < template < auto... > typename Inner, template < typename... > typename Outer, tuple_like auto... Params >
using parametrize_over_combinations_t = parametrize_over_combinations< Inner, Outer, Params... >::type;
} // namespace lstr

#endif // L3STER_UTIL_META_HPP
