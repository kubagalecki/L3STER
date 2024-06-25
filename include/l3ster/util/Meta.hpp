// Meta-programming helper classes

#ifndef L3STER_UTIL_META_HPP
#define L3STER_UTIL_META_HPP

#include "l3ster/util/Common.hpp"
#include "l3ster/util/Concepts.hpp"
#include "l3ster/util/StaticVector.hpp"

#include <functional>
#include <numeric>
#include <utility>
#include <variant>

namespace lstr::util
{
namespace detail
{
template < typename T >
inline constexpr bool is_value_pack = false;
template < auto... V >
inline constexpr bool is_value_pack< ValuePack< V... > > = true;
template < typename T >
inline constexpr bool is_type_pack = false;
template < typename... V >
inline constexpr bool is_type_pack< TypePack< V... > > = true;

template < typename T >
inline constexpr bool has_unique_types = false;
template < typename... Ts >
inline constexpr bool has_unique_types< TypePack< Ts... > > = std::invoke([] {
    constexpr auto get_first_occurrence_index = []< typename T >(TypePack< T >) {
        std::size_t index            = 0;
        const auto  assert_same_type = [&index]< typename Type >(TypePack< Type >) {
            if constexpr (std::same_as< T, Type >)
                return false;
            else
            {
                ++index;
                return true;
            }
        };
        (assert_same_type(TypePack< Ts >{}) and ...);
        return index;
    };
    std::array< bool, sizeof...(Ts) > type_map{};
    const auto                        set_first_occurence = [&](auto t_wrpr) {
        type_map[get_first_occurrence_index(t_wrpr)] = true;
    };
    (set_first_occurence(TypePack< Ts >{}), ...);
    return std::ranges::all_of(type_map, std::identity{});
});
} // namespace detail

template < typename T >
concept ValuePack_c = detail::is_value_pack< T >;
template < typename T >
concept TypePack_c = detail::is_type_pack< T >;
template < typename T >
concept UniqueTypePack_c = TypePack_c< T > and detail::has_unique_types< T >;

namespace detail
{
template < std::integral T, T first, T last >
constexpr auto makeIntervalArray()
    requires(first <= last)
{
    std::array< T, last - first + 1 > interval;
    std::iota(interval.begin(), interval.end(), first);
    return interval;
}

template < std::array A >
constexpr auto intSeqFromArray()
    requires std::integral< typename decltype(A)::value_type >
{
    return []< size_t... I >(std::index_sequence< I... >) {
        return std::integer_sequence< typename decltype(A)::value_type, A[I]... >{};
    }(std::make_index_sequence< A.size() >{});
}

template < std::array A >
struct ArrayToValPackImpl
{
    template < size_t I >
    static constexpr auto access = A[I];
    using type                   = decltype([]< size_t... I >(std::index_sequence< I... >) {
        return ValuePack< access< I >... >{};
    }(std::make_index_sequence< A.size() >{}));
};
} // namespace detail

template < std::integral T, T first, T last >
    requires(first <= last)
using int_seq_interval = decltype(detail::intSeqFromArray< detail::makeIntervalArray< T, first, last >() >());

template < std::array A >
using convert_array_to_value_pack_t = detail::ArrayToValPackImpl< A >::type;

namespace detail
{
template < ValuePack_c >
struct SplitValuePackImpl
{};
template < auto... Vs >
struct SplitValuePackImpl< ValuePack< Vs... > >
{
    using type = TypePack< ValuePack< Vs >... >;
};
} // namespace detail
template < ValuePack_c VP >
using split_value_pack_t = detail::SplitValuePackImpl< VP >::type;

// Functionality related to parametrizing over all combinations of a pack of nttp arrays
namespace detail
{
template < std::array... Arrays >
constexpr inline auto repetitions = std::invoke([] {
    std::array< std::size_t, sizeof...(Arrays) > retval;
    auto                                         push_size = [ins_it = begin(retval)](const auto& array) mutable {
        *ins_it++ = array.size();
    };
    (push_size(Arrays), ...);
    std::exclusive_scan(rbegin(retval), rend(retval), rbegin(retval), 1u, std::multiplies<>{});
    return retval;
});
} // namespace detail

template < std::array... Arrays >
constexpr auto getCartProdComponents()
    requires(sizeof...(Arrays) > 0)
{
    constexpr auto deduction_helper = []< std::size_t... ArrInd >(std::index_sequence< ArrInd... >) {
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
    constexpr auto return_deduction_helper = []< std::size_t... I >(std::index_sequence< I... >) {
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
struct ApplyValuesImpl;
template < template < auto... > typename T, auto... Params >
struct ApplyValuesImpl< T, ValuePack< Params... > >
{
    using type = T< Params... >;
};
} // namespace detail
template < template < auto... > typename T, ValuePack_c Params >
using apply_values_t = typename detail::ApplyValuesImpl< T, Params >::type;

namespace detail
{
template < template < auto... > typename Inner, template < typename... > typename Outer, TypePack_c Params >
struct ApplyInnerOuterImpl;
template < template < auto... > typename Inner, template < typename... > typename Outer, ValuePack_c... Params >
struct ApplyInnerOuterImpl< Inner, Outer, TypePack< Params... > >
{
    using type = Outer< apply_values_t< Inner, Params >... >;
};
} // namespace detail
template < template < auto... > typename Inner, template < typename... > typename Outer, TypePack_c Params >
using apply_in_out_t = typename detail::ApplyInnerOuterImpl< Inner, Outer, Params >::type;

template < template < auto... > typename Inner, template < typename... > typename Outer, std::array... Params >
using cart_prod_t = apply_in_out_t< Inner, Outer, decltype(zipArrays(getCartProdComponents< Params... >())) >;

template < size_t N >
constexpr auto getTrueInds(const std::array< bool, N >& a) -> util::StaticVector< size_t, N >
{
    auto retval = util::StaticVector< size_t, N >{};
    std::ranges::copy_if(std::views::iota(0uz, a.size()), std::back_inserter(retval), [&a](size_t i) { return a[i]; });
    return retval;
}

template < ArrayOf_c< bool > auto A >
constexpr auto getTrueInds(ConstexprValue< A > = {})
{
    constexpr auto true_inds_sv = getTrueInds(A);
    auto           retval       = std::array< size_t, true_inds_sv.size() >{};
    std::ranges::copy(true_inds_sv, retval.begin());
    return retval;
}

template < auto... >
inline constexpr bool always_false = false;
} // namespace lstr::util
#endif // L3STER_UTIL_META_HPP
