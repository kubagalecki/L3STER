#ifndef L3STER_UTIL_ARRAYOWNER
#define L3STER_UTIL_ARRAYOWNER

#include "l3ster/util/Assertion.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <ranges>

namespace lstr::util
{

struct DefaultAllocPolicy
{
    template < class T >
    static constexpr auto alignment = std::align_val_t{alignof(T)};

    template < class T >
    static constexpr T* allocate(std::size_t n)
    {
        return std::allocator< T >{}.allocate(n);
    }
    template < class T >
    static constexpr void deallocate(T* p, std::size_t n) noexcept
    {
        std::allocator< T >{}.deallocate(p, n);
    }
};

template < std::size_t align >
struct AlignedAllocPolicy
{
    static_assert(align != 0 and (align & (align - 1)) == 0, "Alignment must be a power of two");

    template < class T >
    static constexpr auto alignment = std::align_val_t{std::max(align, alignof(T))};

    template < class T >
    static T* allocate(std::size_t n)
    {
        return static_cast< T* >(::operator new(n * sizeof(T), alignment< T >));
    }
    template < class T >
    static void deallocate(T* p, std::size_t n) noexcept
    {
        ::operator delete(p, n * sizeof(T), alignment< T >);
    }
};

template < class P, class T >
concept AllocPolicyFor_c = requires(T* p, std::size_t n) {
    { P::template allocate< T >(n) } -> std::same_as< T* >;
    { P::template deallocate< T >(p, n) } noexcept;
};

/// Fixed-size dynamically allocated array. Exposes a range interface
template < std::default_initializable T, AllocPolicyFor_c< T > AllocPolicy = DefaultAllocPolicy >
class ArrayOwner
{
    static constexpr T* create(std::size_t size)
    {
        auto retval = AllocPolicy::template allocate< T >(size);
        if consteval
        {
            for (std::size_t i = 0; i != size; ++i)
                std::construct_at(retval + i);
        }
        else
        {
            std::uninitialized_default_construct_n(retval, size);
        }
        return retval;
    }
    static constexpr void destroy(T* ptr, std::size_t size)
    {
        if (ptr)
        {
            std::destroy_n(ptr, size);
            AllocPolicy::template deallocate< T >(ptr, size);
        }
    }

public:
    using size_type = std::size_t;

    constexpr ArrayOwner(const ArrayOwner&)            = delete;
    constexpr ArrayOwner& operator=(const ArrayOwner&) = delete;
    constexpr ArrayOwner(ArrayOwner&& other) noexcept : m_data{other.m_data}, m_size{other.m_size}
    {
        other.m_data = nullptr;
    }
    constexpr ArrayOwner& operator=(ArrayOwner&& other) noexcept
    {
        if (this != &other)
        {
            destroy(m_data, m_size);
            m_size       = other.m_size;
            m_data       = other.m_data;
            other.m_data = nullptr;
        }
        return *this;
    }
    constexpr ~ArrayOwner() { destroy(m_data, m_size); }

    constexpr ArrayOwner() = default;
    explicit constexpr ArrayOwner(size_type size) : m_data{create(size)}, m_size{size} {}
    template < std::ranges::range R >
    constexpr ArrayOwner(R&& range) // NOLINT implicit conversion and copy are intended
        requires std::constructible_from< T, std::ranges::range_reference_t< R > >;
    template < std::convertible_to< T > Vals >
    constexpr ArrayOwner(std::initializer_list< Vals > vals);
    constexpr ArrayOwner(size_type size, const T& init) : ArrayOwner(size) { std::ranges::fill(*this, init); }
    template < std::forward_iterator Iterator, std::sentinel_for< Iterator > Sentinel >
    constexpr ArrayOwner(Iterator i, Sentinel s)
        requires std::convertible_to< std::iter_value_t< Iterator >, T >
        : ArrayOwner(std::ranges::subrange(i, s))
    {}

    constexpr T*       begin() { return m_data; }
    constexpr const T* begin() const { return m_data; }
    constexpr const T* cbegin() const { return begin(); }
    constexpr T*       end() { return m_data + m_size; }
    constexpr const T* end() const { return m_data + m_size; }
    constexpr const T* cend() const { return end(); }
    constexpr T*       data() { return m_data; }
    constexpr const T* data() const { return m_data; }
    constexpr T&       operator[](size_type i) { return m_data[i]; }
    constexpr const T& operator[](size_type i) const { return m_data[i]; }
    constexpr T&       at(size_type i, std::source_location sl = std::source_location::current())
    {
        throwingAssert< std::out_of_range >(i < m_size, "ArrayOwner: out of bounds access", sl);
        return m_data[i];
    }
    constexpr const T& at(size_type i, std::source_location sl = std::source_location::current()) const
    {
        throwingAssert< std::out_of_range >(i < m_size, "ArrayOwner: out of bounds access", sl);
        return m_data[i];
    }
    constexpr T&        front() { return m_data[0]; }
    constexpr const T&  front() const { return m_data[0]; }
    constexpr T&        back() { return m_data[m_size - 1]; }
    constexpr const T&  back() const { return m_data[m_size - 1]; }
    constexpr size_type size() const { return m_size; }
    constexpr bool      empty() const { return m_size == 0; }

    friend constexpr auto operator<=>(const ArrayOwner& a, const ArrayOwner& b)
    {
        return std::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end());
    }
    friend constexpr bool operator==(const ArrayOwner& a, const ArrayOwner& b) { return std::ranges::equal(a, b); }

private:
    T*        m_data{};
    size_type m_size{};
};
template < std::ranges::range R >
ArrayOwner(R&&) -> ArrayOwner< std::ranges::range_value_t< R > >;
template < typename T >
ArrayOwner(std::initializer_list< T >) -> ArrayOwner< T >;

template < std::default_initializable T, AllocPolicyFor_c< T > AllocPolicy >
template < std::ranges::range R >
constexpr ArrayOwner< T, AllocPolicy >::ArrayOwner(R&& range)
    requires std::constructible_from< T, std::ranges::range_reference_t< R > >
    : ArrayOwner(static_cast< size_t >(std::ranges::distance(range)))
{
    std::ranges::copy(std::forward< R >(range), begin());
}

template < std::default_initializable T, AllocPolicyFor_c< T > AllocPolicy >
template < std::convertible_to< T > Vals >
constexpr ArrayOwner< T, AllocPolicy >::ArrayOwner(std::initializer_list< Vals > vals) : ArrayOwner(vals.size())
{
    std::move(vals.begin(), vals.end(), begin());
}

template < typename T >
constexpr auto copy(const ArrayOwner< T >& a)
{
    auto retval = ArrayOwner< T >(a.size());
    std::ranges::copy(a, retval.begin());
    return retval;
}
} // namespace lstr::util
#endif // L3STER_UTIL_ARRAYOWNER