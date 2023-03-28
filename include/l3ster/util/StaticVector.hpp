#ifndef L3STER_STATICVECTOR_HPP
#define L3STER_STATICVECTOR_HPP

#include <array>

namespace lstr::detail
{
template < typename T, std::size_t capacity >
class StaticVector
{
public:
    using value_type = T;

    T*       begin() { return m_data.begin(); }
    const T* begin() const { return m_data.begin(); }
    T*       end() { return m_data.begin() + m_size; }
    const T* end() const { return m_data.begin() + m_size; }

    T*       data() { return m_data.data(); }
    const T* data() const { return m_data.data(); }

    T&       operator[](std::size_t i) { return m_data[i]; }
    const T& operator[](std::size_t i) const { return m_data[i]; }
    T&       front() { return m_data[0]; }
    const T& front() const { return m_data[0]; }
    T&       back() { return m_data[m_size - 1]; }
    const T& back() const { return m_data[m_size - 1]; }

    [[nodiscard]] std::size_t size() const { return m_size; }

    void push_back(T t) { m_data[m_size++] = std::move(t); }
    void pop_back() { --m_size; }

private:
    std::array< T, capacity > m_data;
    std::size_t               m_size{};
};
} // namespace lstr::detail
#endif // L3STER_STATICVECTOR_HPP
