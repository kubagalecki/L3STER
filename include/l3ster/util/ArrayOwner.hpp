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
    explicit ArrayOwner(std::size_t size) : m_data{std::make_unique_for_overwrite< T[] >(size)}, m_size{size} {}

    T*          begin() { return m_data.get(); }
    const T*    begin() const { return m_data.get(); }
    T*          end() { return m_data.get() + m_size; }
    const T*    end() const { return m_data.get() + m_size; }
    T*          data() { return m_data.get(); }
    const T*    data() const { return m_data.get(); }
    T&          operator[](std::size_t i) { return m_data[i]; }
    const T&    operator[](std::size_t i) const { return m_data[i]; }
    T&          front() { return m_data[0]; }
    const T&    front() const { return m_data[0]; }
    T&          back() { return m_data[m_size - 1]; }
    const T&    back() const { return m_data[m_size - 1]; }
    std::size_t size() const { return m_size; }

private:
    std::unique_ptr< T[] > m_data;
    std::size_t            m_size{};
};
} // namespace lstr::util
#endif // L3STER_UTIL_ARRAYOWNER