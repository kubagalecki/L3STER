#ifndef L3STER_STATICVECTOR_HPP
#define L3STER_STATICVECTOR_HPP

#include "l3ster/util/Common.hpp"

namespace lstr::util
{
template < std::default_initializable T, std::size_t capacity >
class StaticVector
{
public:
    using value_type = T;
    using size_type  = smallest_integral_t< capacity >;

    StaticVector() = default;
    template < size_t N >
    StaticVector(const std::array< T, N >& a)
        requires(N <= capacity)
        : m_size{N}
    {
        std::ranges::copy(a, m_data.begin());
    }
    template < std::ranges::range R >
    explicit StaticVector(R&& r)
        requires std::convertible_to< std::ranges::range_value_t< R >, T >
    {
        std::ranges::copy(std::forward< R >(r), std::back_inserter(*this));
    }
    template < std::ranges::sized_range R >
    explicit StaticVector(R&& r)
        requires std::convertible_to< std::ranges::range_value_t< R >, T >
        : m_size{static_cast< size_type >(std::ranges::size(r))}
    {
        std::ranges::copy(std::forward< R >(r), m_data.begin());
    }

    constexpr T*       begin() { return m_data.begin(); }
    constexpr const T* begin() const { return m_data.begin(); }
    constexpr T*       end() { return m_data.begin() + m_size; }
    constexpr const T* end() const { return m_data.begin() + m_size; }

    constexpr T*       data() { return m_data.data(); }
    constexpr const T* data() const { return m_data.data(); }

    constexpr T&       operator[](std::size_t i) { return m_data[i]; }
    constexpr const T& operator[](std::size_t i) const { return m_data[i]; }
    constexpr T&       front() { return m_data[0]; }
    constexpr const T& front() const { return m_data[0]; }
    constexpr T&       back() { return m_data[m_size - 1]; }
    constexpr const T& back() const { return m_data[m_size - 1]; }

    [[nodiscard]] constexpr std::size_t size() const { return m_size; }
    [[nodiscard]] constexpr bool        empty() const { return m_size == 0; }

    constexpr void resize(size_type size, const T& val = T{})
    {
        if (size <= m_size)
            erase(std::prev(end(), m_size - size), end());
        else
        {
            std::fill(end(), begin() + size, val);
            m_size = size;
        }
    }
    constexpr void push_back(T t)
    {
        util::throwingAssert< std::bad_alloc >(m_size < capacity);
        m_data[m_size++] = std::move(t);
    }
    constexpr void pop_back() { std::destroy_at(begin() + --m_size); }
    constexpr T*   erase(const T* first, const T* last)
    {
        const auto last_moved = std::move(const_cast< T* >(last), end(), const_cast< T* >(first));
        std::destroy(last_moved, end());
        m_size -= static_cast< size_type >(std::distance(first, last));
        return const_cast< T* >(last);
    }

private:
    std::array< T, capacity > m_data;
    size_type                 m_size{};
};

template < typename T, size_t N1, size_t N2 >
auto operator<=>(const StaticVector< T, N1 >& v1, const StaticVector< T, N2 >& v2)
{
    return std::lexicographical_compare_three_way(v1.begin(), v1.end(), v2.begin(), v2.end());
}
template < typename T, size_t N1, size_t N2 >
auto operator==(const StaticVector< T, N1 >& v1, const StaticVector< T, N2 >& v2)
{
    return std::ranges::equal(v1, v2);
}
} // namespace lstr::util
#endif // L3STER_STATICVECTOR_HPP
