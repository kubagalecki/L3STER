// Meta-programming helper classes

#ifndef L3STER_UTIL_META_HPP
#define L3STER_UTIL_META_HPP

#include "l3ster/util/Common.hpp"
#include "l3ster/util/Concepts.hpp"

#include <numeric>
#include <utility>
#include <variant>

namespace lstr
{
template < auto V >
struct ConstexprValue
{
    using type                  = decltype(V);
    static constexpr auto value = V;
};

template < typename... Types >
struct TypePack
{
    static constexpr auto size = sizeof...(Types);
};

template < auto... Vals >
struct ValuePack
{
    static constexpr auto size = sizeof...(Vals);
};

namespace detail
{
template < typename T >
struct IsValuePack : std::false_type
{};
template < auto... V >
struct IsValuePack< ValuePack< V... > > : std::true_type
{};

template < typename T >
struct IsTypePack : std::false_type
{};
template < typename... V >
struct IsTypePack< TypePack< V... > > : std::true_type
{};
} // namespace detail

template < typename T >
concept ValuePack_c = detail::IsValuePack< T >::value;
template < typename T >
concept TypePack_c = detail::IsTypePack< T >::value;

namespace detail
{
template < std::integral T, T first, T last >
    requires(first <= last)
constexpr auto make_interval_array()
{
    std::array< T, last - first + 1 > interval;
    std::iota(interval.begin(), interval.end(), first);
    return interval;
}

template < std::array A >
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
    requires(first <= last)
using int_seq_interval = decltype(detail::int_seq_from_array< detail::make_interval_array< T, first, last >() >());

namespace detail
{
template < typename T >
struct Constify
{
    const T operator()(const T& in)
        requires(not std::is_pointer_v< T >)
    {
        return in;
    }
    const std::pointer_traits< T >::element_type* operator()(T in)
        requires(std::is_pointer_v< T >)
    {
        return in;
    }
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

template < typename T, T... V >
constexpr auto makeArrayFromValueSet(ValuePack< V... >)
{
    return std::array{V...};
}

namespace detail
{
template < std::array... Arrays >
constexpr inline auto repetitions = [] {
    std::array< std::size_t, sizeof...(Arrays) > retval;
    auto                                         push_size = [ins_it = begin(retval)](const auto& array) mutable {
        *ins_it++ = array.size();
    };
    (push_size(Arrays), ...);
    std::exclusive_scan(rbegin(retval), rend(retval), rbegin(retval), 1u, std::multiplies<>{});
    return retval;
}();
} // namespace detail

template < std::array... Arrays >
    requires(sizeof...(Arrays) > 0)
constexpr auto getCartProdComponents()
{
    constexpr auto deduction_helper = []< std::size_t... ArrInd >(std::index_sequence< ArrInd... >)
    {
        constexpr auto repeat_by_ind = []< std::size_t Ind >(std::integral_constant< std::size_t, Ind >) {
            constexpr auto repeat = [](const auto& array, std::size_t pack_ind) {
                constexpr auto prod_size = (Arrays.size() * ...);
                std::array< typename std::decay_t< decltype(array) >::value_type, prod_size > retval;
                for (auto ins_it = begin(retval); ins_it != end(retval);)
                    for (auto v : array)
                        for ([[maybe_unused]] auto i : std::views::iota(0u, detail::repetitions< Arrays... >[pack_ind]))
                            *ins_it++ = v;
                return retval;
            };
            constexpr auto arr_refs = std::tie(Arrays...);
            return repeat(std::get< Ind >(arr_refs), Ind);
        };
        return ValuePack< repeat_by_ind(std::integral_constant< std::size_t, ArrInd >{})... >{};
    };
    return deduction_helper(std::make_index_sequence< sizeof...(Arrays) >{});
}

namespace detail
{
template < typename... T, std::size_t N >
constexpr std::size_t deduceArraySizes(const std::array< T, N >&...)
{
    return N;
}
} // namespace detail

template < std::array... V >
auto zipArrays(ValuePack< V... >)
{
    constexpr auto return_deduction_helper = []< std::size_t... I >(std::index_sequence< I... >)
    {
        constexpr auto make_nth_entry = []< std::size_t N >(std::integral_constant< std::size_t, N >) {
            return ValuePack< V[N]... >{};
        };
        return TypePack< decltype(make_nth_entry(std::integral_constant< std::size_t, I >{}))... >{};
    };
    return return_deduction_helper(std::make_index_sequence< detail::deduceArraySizes(V...) >{});
}

namespace detail
{
template < template < auto... > typename T, ValuePack_c Params >
struct ApplyValuesDeductionHelper;
template < template < auto... > typename T, auto... Params >
struct ApplyValuesDeductionHelper< T, ValuePack< Params... > >
{
    using type = T< Params... >;
};
} // namespace detail
template < template < auto... > typename T, ValuePack_c Params >
using apply_values_t = typename detail::ApplyValuesDeductionHelper< T, Params >::type;

namespace detail
{
template < template < auto... > typename Inner, template < typename... > typename Outer, TypePack_c Params >
struct ApplyInnerOuterDeductionHelper;
template < template < auto... > typename Inner, template < typename... > typename Outer, ValuePack_c... Params >
struct ApplyInnerOuterDeductionHelper< Inner, Outer, TypePack< Params... > >
{
    using type = Outer< apply_values_t< Inner, Params >... >;
};
} // namespace detail
template < template < auto... > typename Inner, template < typename... > typename Outer, TypePack_c Params >
using apply_in_out_t = typename detail::ApplyInnerOuterDeductionHelper< Inner, Outer, Params >::type;

template < template < auto... > typename Inner, template < typename... > typename Outer, std::array... Params >
using cart_prod_t = apply_in_out_t< Inner, Outer, decltype(zipArrays(getCartProdComponents< Params... >())) >;

template < array_of< bool > auto A >
consteval auto getTrueInds()
{
    std::array< size_t, std::ranges::count(A, true) > retval;
    auto                                              insert_it = retval.begin();
    for (size_t i = 0; bool v : A)
    {
        if (v)
            *insert_it++ = i;
        ++i;
    }
    return retval;
}
} // namespace lstr
#endif // L3STER_UTIL_META_HPP
