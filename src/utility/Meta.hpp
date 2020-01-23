#ifndef L3STER_INCGUARD_UTIL_META_HPP
#define L3STER_INCGUARD_UTIL_META_HPP

#include <array>
#include <tuple>
#include <utility>

namespace lstr::util::meta
{

// Helper class - determines if T is a static array or std::array type
template <typename T>
struct is_array : std::false_type {};

template <typename T, std::size_t N>
struct is_array<T[N]> : std::true_type {};

template <typename T, std::size_t N>
struct is_array< const std::array<T, N> > : std::true_type {};

template <typename T>
struct array;

template <typename T, std::size_t N>
struct array<T[N]>
{
    static constexpr size_t size = N;
    using type = T;
};

template <typename T, std::size_t N>
struct array< const std::array<T, N> >
{
    static constexpr size_t size = N;
    using type = T;
};

// lift constexpr array to std::integer_sequence
// ArrayClass should have a public static constexpr member 'array'
template <typename, typename>
struct convert_to_int_seq;

template <typename ArrayClass, size_t ... Ints>
struct convert_to_int_seq< ArrayClass, std::index_sequence<Ints ...> >
{
    using type = std::integer_sequence<typename array<decltype(ArrayClass::values)>::type,
          ArrayClass::values[Ints] ...>;
};

template <typename ArrayClass>
struct array_to_intseq
{
    static_assert(is_array<decltype(ArrayClass::values)>::value,
                  "ArrayClass template parameter must have member 'array'");

    using type = typename convert_to_int_seq< ArrayClass,
          std::make_index_sequence< array<decltype(ArrayClass::values)>::size > >::type;
};

// Get nth element of std::integer_sequence
template <typename, auto>
struct get_intseq_n;

template <auto I, typename T, T ... Ints>
struct get_intseq_n< std::integer_sequence<T, Ints ...>, I >
{
    static constexpr T value = std::array<T, sizeof...(Ints)> {Ints ...} [I];
};

// convert std::integer_sequence to constexpr array
template <typename T, T ... Ints>
constexpr auto intseq_to_array(std::integer_sequence<T, Ints ...>)
{
    return std::array<T, sizeof...(Ints)> {Ints ...};
}

// Add value to std::integer_sequence
template <auto, typename>
struct add_to_intseq;

template <auto I, typename T, T ... Ints>
struct add_to_intseq< I, std::integer_sequence<T, Ints ...> >
{
    using type = std::integer_sequence<T, Ints ..., I>;
};

// Repeat value to form std::integer_sequence
template <auto I, size_t N>
struct rep_int
{
    using type =
        typename add_to_intseq < I, typename rep_int < I, N - 1 >::type >::type;
};

template <auto I>
struct rep_int<I, 0>
{
    using type = std::integer_sequence<decltype(I)>;
};

// Concatenate multiple std::integer_sequence into one
template <typename ...>
struct cat_intseq;

template <typename T, T ... Ints>
struct cat_intseq< std::integer_sequence<T, Ints ...> >
{
    using type = std::integer_sequence<T, Ints ...>;
};

template <typename T, typename U, T ... Ints>
struct cat_intseq< std::integer_sequence<T, Ints ...>, std::integer_sequence<U> >
{
    static_assert(std::is_same_v<T, U>, "Concatenated integer sequences must share integer type");
    using type = std::integer_sequence<T, Ints ...>;
};

template <typename T, typename U, U Ints>
struct cat_intseq< T, std::integer_sequence<U, Ints> >
{
    using type = typename add_to_intseq<Ints, T>::type;
};

template <typename T, typename U, U Int1, U ... Ints>
struct cat_intseq< T, std::integer_sequence<U, Int1, Ints ...> >
{
    using type = typename cat_intseq< typename add_to_intseq<Int1, T>::type,
          std::integer_sequence<U, Ints ...> >::type;
};

template <typename T, typename U, typename ... V>
struct cat_intseq<T, U, V ...>
{
    using type = typename cat_intseq< typename cat_intseq<T, U>::type, V ...>::type;
};

// Stretch std::integer_sequence
template <typename, size_t N>
struct stretch_intseq;

template <typename T, T ... Ints, size_t N>
struct stretch_intseq< std::integer_sequence<T, Ints ...>, N >
{
    using type =
        typename cat_intseq< typename rep_int<Ints, N>::type ... >::type;
};

// Repeat std::integer_sequence N times
template <typename T, size_t N>
struct repeat_intseq
{
    using type =
        typename cat_intseq < typename repeat_intseq < T, N - 1 >::type, T >::type;
};

template <typename T, T ... Ints>
struct repeat_intseq< std::integer_sequence<T, Ints ...>, 0 >
{
    using type = std::integer_sequence<T>;
};

// Repeat std::integer_sequence of length N N times
template <typename>
struct rep_intseq;

template <typename T, T ... Ints>
struct rep_intseq< std::integer_sequence<T, Ints ...> >
{
    using type = typename repeat_intseq< std::integer_sequence<T, Ints ...>,
          sizeof...(Ints) >::type;
};

template < template <typename ...> typename,
           template <auto, auto> typename, typename, typename, typename>
struct combine2_helper;

template < template <typename ...> typename M, template <auto, auto> typename C,
           typename I1, typename I2, typename D, D ... Ints>
struct combine2_helper< M, C, I1, I2, std::integer_sequence<D, Ints ...> >
{
    using type =
        M< C< get_intseq_n<I1, Ints>::value, get_intseq_n<I2, Ints>::value > ... >;
};

template < template <typename ...> typename M,
           template <auto, auto> typename C, typename I1, typename I2>
struct combine2
{
    using type = typename combine2_helper< M, C, I1, I2,
          std::make_index_sequence<I1::size()> >::type;
};

/////////////////////////////////////////////////////////////////////////////
template < template <typename ...> typename M,
           template <auto, auto> typename C, typename I1, typename I2>
struct cart2
{
private:
    using I1_i = typename array_to_intseq<I1>::type;
    using I2_i = typename array_to_intseq<I2>::type;
public:
    using type = typename combine2< M, C,
          typename stretch_intseq<I1_i, I2_i::size()>::type,
          typename repeat_intseq<I2_i, I1_i::size()>::type >::type;
};
/////////////////////////////////////////////////////////////////////////////

}               // namespace lstr::util::meta

#endif          // end include guard
