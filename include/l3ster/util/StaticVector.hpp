#ifndef L3STER_STATICVECTOR_HPP
#define L3STER_STATICVECTOR_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <memory>

namespace lstr::util
{
template < std::default_initializable T, std::size_t capacity >
class StaticVector
{
public:
    using value_type = T;

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

    constexpr void resize(std::size_t size, const T& val = T{})
    {
        if (size <= m_size)
            erase(std::prev(end(), m_size - size), end());
        else
        {
            std::fill(end(), begin() + size, val);
            m_size = size;
        }
    }
    constexpr void push_back(T t) { m_data[m_size++] = std::move(t); }
    constexpr void pop_back() { std::destroy_at(begin() + --m_size); }
    constexpr T*   erase(const T* first, const T* last)
    {
        const auto last_moved = std::move(const_cast< T* >(last), end(), const_cast< T* >(first));
        std::destroy(last_moved, end());
        m_size -= static_cast< std::size_t >(std::distance(first, last));
        return const_cast< T* >(last);
    }

private:
    std::array< T, capacity > m_data;
    std::size_t               m_size{};
};
} // namespace lstr::util
#endif // L3STER_STATICVECTOR_HPP
