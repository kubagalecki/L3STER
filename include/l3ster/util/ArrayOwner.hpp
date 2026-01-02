#ifndef L3STER_UTIL_ARRAYOWNER
#define L3STER_UTIL_ARRAYOWNER

#include "l3ster/util/Assertion.hpp"

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <memory>

namespace lstr::util
{
struct CacheAlign
{};
inline constexpr CacheAlign cache_align{};

/// Fixed-size dynamically allocated array. Exposed a range interface
template < std::default_initializable T >
class ArrayOwner
{
    static constexpr std::size_t cacheline_size = std::hardware_destructive_interference_size;

public:
    using size_type = std::size_t;
    ArrayOwner()    = default;
    explicit ArrayOwner(size_type size) : m_size{size}, m_data{std::make_unique_for_overwrite< T[] >(size)} {}
    ArrayOwner(size_type size, std::align_val_t align) : m_size{size}, m_data{new(align) T[size]} {}
    ArrayOwner(size_type size, CacheAlign) : ArrayOwner(size, std::align_val_t{std::max(cacheline_size, alignof(T))}) {}
    template < std::ranges::range R >
    ArrayOwner(R&& range) // NOLINT implicit conversion and copy are intended
        requires std::constructible_from< T, std::ranges::range_reference_t< R > >;
    template < std::convertible_to< T > Vals >
    ArrayOwner(std::initializer_list< Vals > vals);
    ArrayOwner(size_type size, const T& init) : ArrayOwner(size) { std::ranges::fill(*this, init); }
    template < std::forward_iterator Iterator, std::sentinel_for< Iterator > Sentinel >
    ArrayOwner(Iterator i, Sentinel s)
        requires std::convertible_to< std::iter_value_t< Iterator >, T >
        : ArrayOwner(std::ranges::subrange(i, s))
    {}

    T*       begin() { return m_data.get(); }
    const T* begin() const { return m_data.get(); }
    const T* cbegin() const { return begin(); }
    T*       end() { return m_data.get() + m_size; }
    const T* end() const { return m_data.get() + m_size; }
    const T* cend() const { return end(); }
    T*       data() { return m_data.get(); }
    const T* data() const { return m_data.get(); }
    T&       operator[](size_type i) { return m_data[i]; }
    const T& operator[](size_type i) const { return m_data[i]; }
    T&       at(size_type i, std::source_location sl = std::source_location::current())
    {
        util::throwingAssert(i < m_size, "ArrayOwner: out of bounds access", sl);
        return m_data[i];
    }
    const T& at(size_type i, std::source_location sl = std::source_location::current()) const
    {
        util::throwingAssert(i < m_size, "ArrayOwner: out of bounds access", sl);
        return m_data[i];
    }
    T&        front() { return m_data[0]; }
    const T&  front() const { return m_data[0]; }
    T&        back() { return m_data[m_size - 1]; }
    const T&  back() const { return m_data[m_size - 1]; }
    size_type size() const { return m_size; }
    bool      empty() const { return m_size == 0; }

    friend auto operator<=>(const ArrayOwner& a, const ArrayOwner< T >& b)
    {
        return std::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end());
    }
    friend bool operator==(const ArrayOwner& a, const ArrayOwner< T >& b) { return std::ranges::equal(a, b); }

private:
    size_type              m_size{};
    std::unique_ptr< T[] > m_data;
};
template < std::ranges::range R >
ArrayOwner(R&&) -> ArrayOwner< std::ranges::range_value_t< R > >;
template < typename T >
ArrayOwner(std::initializer_list< T >) -> ArrayOwner< T >;

template < std::default_initializable T >
template < std::ranges::range R >
ArrayOwner< T >::ArrayOwner(R&& range)
    requires std::constructible_from< T, std::ranges::range_reference_t< R > >
    : m_size{static_cast< size_t >(std::ranges::distance(range))}, m_data{std::make_unique_for_overwrite< T[] >(m_size)}
{
    std::ranges::copy(std::forward< R >(range), begin());
}

template < std::default_initializable T >
template < std::convertible_to< T > Vals >
ArrayOwner< T >::ArrayOwner(std::initializer_list< Vals > vals)
    : m_size{vals.size()}, m_data{std::make_unique_for_overwrite< T[] >(m_size)}
{
    std::move(vals.begin(), vals.end(), begin());
}

template < typename T >
auto copy(const ArrayOwner< T >& a)
{
    auto retval = ArrayOwner< T >(a.size());
    std::ranges::copy(a, retval.begin());
    return retval;
}
} // namespace lstr::util
#endif // L3STER_UTIL_ARRAYOWNER