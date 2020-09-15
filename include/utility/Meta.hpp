// Meta-programming helper classes

#ifndef L3STER_INCGUARD_UTIL_META_HPP
#define L3STER_INCGUARD_UTIL_META_HPP

#include <array>
#include <tuple>
#include <utility>

/*Comments:
 *  naming convention conforming to the standard library
 *  value_sequence introduced because std::integer_sequence does not support enums (in msvc)
 */

namespace lstr::util::meta
{

// Helper class - determines if T is a static array or std::array type
template < typename T >
struct is_array : std::false_type
{};

template < typename T, std::size_t N >
struct is_array< T[N] > : std::true_type
{};

template < typename T, std::size_t N >
struct is_array< const std::array< T, N > > : std::true_type
{};

template < typename T >
struct array;

template < typename T, std::size_t N >
struct array< T[N] >
{
    static constexpr size_t size = N;
    using type                   = T;
};

template < typename T, std::size_t N >
struct array< const std::array< T, N > >
{
    static constexpr size_t size = N;
    using type                   = T;
};

// This class is essentially std::integer_sequence, but extended to all NTTP
template < typename T, T... Vals >
struct value_sequence
{
    using value_type = T;

    static constexpr size_t size = sizeof...(Vals);
};

// lift constexpr array to value_sequence
// ArrayClass should have a public static constexpr member 'values'
template < typename, typename >
struct convert_to_val_seq;

template < typename ArrayClass, size_t... Vals >
struct convert_to_val_seq< ArrayClass, std::index_sequence< Vals... > >
{
    using type = value_sequence< typename array< decltype(ArrayClass::values) >::type,
                                 ArrayClass::values[Vals]... >;
};

template < typename ArrayClass >
struct array_to_valseq
{
    static_assert(is_array< decltype(ArrayClass::values) >::value,
                  "ArrayClass template parameter must have member 'array'");

    using type = typename convert_to_val_seq<
        ArrayClass,
        std::make_index_sequence< array< decltype(ArrayClass::values) >::size > >::type;
};

// Get nth element of value_sequence
template < typename, auto >
struct get_valseq_n;

template < size_t I, typename T, T... Vals >
struct get_valseq_n< value_sequence< T, Vals... >, I >
{
    static constexpr T value = std::array< T, sizeof...(Vals) >{Vals...}[I];
};

// convert value_sequence to constexpr array
template < typename T, T... Vals >
constexpr auto valseq_to_array(value_sequence< T, Vals... >)
{
    return std::array< T, sizeof...(Vals) >{Vals...};
}

// Add value to value_sequence
template < auto, typename >
struct add_to_valseq;

template < auto I, typename T, T... Vals >
struct add_to_valseq< I, value_sequence< T, Vals... > >
{
    using type = value_sequence< T, Vals..., I >;
};

// Repeat value to form value_sequence
template < auto I, size_t N >
struct rep_val
{
    using type = typename add_to_valseq< I, typename rep_val< I, N - 1 >::type >::type;
};

template < auto I >
struct rep_val< I, 0 >
{
    using type = value_sequence< decltype(I) >;
};

// Concatenate multiple value_sequence into one
template < typename... >
struct cat_valseq;

template < typename T, T... Vals >
struct cat_valseq< value_sequence< T, Vals... > >
{
    using type = value_sequence< T, Vals... >;
};

template < typename T, typename U, T... Vals >
struct cat_valseq< value_sequence< T, Vals... >, value_sequence< U > >
{
    static_assert(std::is_same_v< T, U >, "Concatenated integer sequences must share integer type");
    using type = value_sequence< T, Vals... >;
};

template < typename T, typename U, U Vals >
struct cat_valseq< T, value_sequence< U, Vals > >
{
    using type = typename add_to_valseq< Vals, T >::type;
};

template < typename T, typename U, U Val1, U... Vals >
struct cat_valseq< T, value_sequence< U, Val1, Vals... > >
{
    using type = typename cat_valseq< typename add_to_valseq< Val1, T >::type,
                                      value_sequence< U, Vals... > >::type;
};

template < typename T, typename U, typename... V >
struct cat_valseq< T, U, V... >
{
    using type = typename cat_valseq< typename cat_valseq< T, U >::type, V... >::type;
};

// Stretch value_sequence
template < typename, size_t N >
struct stretch_valseq;

template < typename T, T... Vals, size_t N >
struct stretch_valseq< value_sequence< T, Vals... >, N >
{
    using type = typename cat_valseq< typename rep_val< Vals, N >::type... >::type;
};

// Repeat value_sequence N times
template < typename T, size_t N >
struct repeat_valseq
{
    using type = typename cat_valseq< typename repeat_valseq< T, N - 1 >::type, T >::type;
};

template < typename T, T... Vals >
struct repeat_valseq< value_sequence< T, Vals... >, 0 >
{
    using type = value_sequence< T >;
};

// Repeat value_sequence of length N N times
template < typename >
struct rep_valseq;

template < typename T, T... Vals >
struct rep_valseq< value_sequence< T, Vals... > >
{
    using type = typename repeat_valseq< value_sequence< T, Vals... >, sizeof...(Vals) >::type;
};

template < template < typename... > typename,
           template < auto, auto >
           typename,
           typename,
           typename,
           typename >
struct combine2_helper;

template < template < typename... > typename M,
           template < auto, auto >
           typename C,
           typename I1,
           typename I2,
           typename D,
           D... Ints >
struct combine2_helper< M, C, I1, I2, std::integer_sequence< D, Ints... > >
{
    using type = M< C< get_valseq_n< I1, Ints >::value, get_valseq_n< I2, Ints >::value >... >;
};

template < template < typename... > typename M,
           template < auto, auto >
           typename C,
           typename I1,
           typename I2 >
struct combine2
{
    using type =
        typename combine2_helper< M, C, I1, I2, std::make_index_sequence< I1::size > >::type;
};

template < template < typename... > typename T, template < auto > typename U, typename A >
struct apply_valseq;

template < template < typename... > typename M,
           template < auto, auto >
           typename C,
           typename I1,
           typename I2 >
struct cartesian_product
{
private:
    using I1_i = typename array_to_valseq< I1 >::type;
    using I2_i = typename array_to_valseq< I2 >::type;

public:
    using type = typename combine2< M,
                                    C,
                                    typename stretch_valseq< I1_i, I2_i::size >::type,
                                    typename repeat_valseq< I2_i, I1_i::size >::type >::type;
};

template < typename... V >
struct and_pack
{
    static constexpr bool value = (V::value && ...);
};

/////////////////////////////////////////////////////////////////////////////
template < template < typename... > typename M,
           template < auto, auto >
           typename C,
           typename I1,
           typename I2 >
using cartesian_product_t = typename cartesian_product< M, C, I1, I2 >::type;

template < template < typename... > typename T,
           template < auto >
           typename U,
           typename A_t,
           A_t... vals >
struct apply_valseq< T, U, value_sequence< A_t, vals... > >
{
    using type = T< U< vals >... >;
};

template < typename... V >
inline constexpr bool and_pack_v = and_pack< V... >::value;
/////////////////////////////////////////////////////////////////////////////

} // namespace lstr::util::meta

#endif // end include guard
