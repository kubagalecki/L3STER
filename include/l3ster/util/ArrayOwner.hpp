#ifndef L3STER_UTIL_ARRAYOWNER
#define L3STER_UTIL_ARRAYOWNER

#include <concepts>
#include <cstdint>
#include <memory>

namespace lstr::util
{
template < std::default_initializable T >
class ArrayOwner
{
public:
    ArrayOwner() = default;
    explicit ArrayOwner(std::size_t size) : m_size{size}, m_data{std::make_unique_for_overwrite< T[] >(size)} {}
    template < std::ranges::range R >
    ArrayOwner(R&& range) // NOLINT implicit conversion and copy are intended
        requires std::constructible_from< T, std::ranges::range_reference_t< R > >;
    template < std::convertible_to< T > Vals >
    ArrayOwner(std::initializer_list< Vals > vals);

    T*       begin() { return m_data.get(); }
    const T* begin() const { return m_data.get(); }
    T*       end() { return m_data.get() + m_size; }
    const T* end() const { return m_data.get() + m_size; }
    T*       data() { return m_data.get(); }
    const T* data() const { return m_data.get(); }
    T&       operator[](std::size_t i) { return m_data[i]; }
    const T& operator[](std::size_t i) const { return m_data[i]; }
    T&       at(std::size_t i)
    {
        assertInRange(i);
        return m_data[i];
    }
    const T& at(std::size_t i) const
    {
        assertInRange(i);
        return m_data[i];
    }
    T&          front() { return m_data[0]; }
    const T&    front() const { return m_data[0]; }
    T&          back() { return m_data[m_size - 1]; }
    const T&    back() const { return m_data[m_size - 1]; }
    std::size_t size() const { return m_size; }
    bool        empty() const { return m_size == 0; }

    friend auto operator<=>(const ArrayOwner& a, const ArrayOwner< T >& b)
    {
        return std::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end());
    }
    friend bool operator==(const ArrayOwner& a, const ArrayOwner< T >& b) { return std::ranges::equal(a, b); }

private:
    void assertInRange(std::size_t i) const
    {
        if (i >= m_size)
            throw std::out_of_range{"Out of bounds access to ArrayOwner"};
    }

    std::size_t            m_size{};
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