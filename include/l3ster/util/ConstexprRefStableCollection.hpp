#ifndef L3STER_CONSTEXPRREFSTABLECOLLECTION_HPP
#define L3STER_CONSTEXPRREFSTABLECOLLECTION_HPP

#include "l3ster/util/ConstexprVector.hpp"

#include <compare>

namespace lstr::util
{
template < typename T >
class ConstexprUniquePtr
{
public:
    constexpr ConstexprUniquePtr() = default;
    constexpr ConstexprUniquePtr(T* ptr) : m_ptr{ptr} {}
    constexpr ConstexprUniquePtr(const ConstexprUniquePtr&)            = delete;
    constexpr ConstexprUniquePtr& operator=(const ConstexprUniquePtr&) = delete;
    constexpr ConstexprUniquePtr(ConstexprUniquePtr&& other) noexcept : m_ptr{std::exchange(other.m_ptr, nullptr)} {}
    constexpr ConstexprUniquePtr& operator=(ConstexprUniquePtr&& other) noexcept
    {
        if (this != std::addressof(other))
            m_ptr = std::exchange(other.m_ptr, nullptr);
        return *this;
    }
    constexpr ~ConstexprUniquePtr()
    {
        if (m_ptr)
        {
            std::destroy_at(m_ptr);
            std::allocator< T >{}.deallocate(m_ptr, 1);
        }
    }

    constexpr T& operator*() const { return *m_ptr; }
    constexpr T* operator->() const { return m_ptr; }
    constexpr T* get() const { return m_ptr; }

private:
    T* m_ptr = nullptr;
};

template < typename T, typename... Args >
constexpr ConstexprUniquePtr< T > constexprMakeUnique(Args&&... args)
    requires std::constructible_from< T, Args... >
{
    const auto alloc = std::allocator< T >{}.allocate(1);
    std::construct_at(alloc, std::forward< Args >(args)...);
    return ConstexprUniquePtr< T >(alloc);
}

template < typename T, size_t BS = 64 >
class ConstexprRefStableCollection
{
public:
    static constexpr size_t block_size = BS;

private:
    using block_t = std::array< T, block_size >;
    static constexpr std::array< size_t, 2 > getInds(size_t pos) { return {pos / block_size, pos % block_size}; }

public:
    class Iterator
    {
        friend class ConstexprRefStableCollection;
        constexpr Iterator(ConstexprUniquePtr< block_t >* bb, ptrdiff_t i) : m_blocks_begin{bb}, m_index{i} {}

    public:
        using difference_type   = ptrdiff_t;
        using value_type        = T;
        using pointer           = T*;
        using reference         = T&;
        using iterator_category = std::random_access_iterator_tag;
        using container_type    = ConstexprRefStableCollection< T, block_size >;

        constexpr Iterator() = default;

        constexpr T& operator[](difference_type ind) const
        {
            const auto [block_ind, el_pos] = getInds(m_index + ind);
            return m_blocks_begin[block_ind]->operator[](el_pos);
        }
        constexpr T& operator*() const { return this->operator[](0); }
        constexpr T* operator->() const { return std::addressof(**this); }

        constexpr auto operator==(const Iterator& other) const { return m_index == other.m_index; }
        constexpr auto operator<=>(const Iterator& other) const { return m_index <=> other.m_index; }

        constexpr Iterator  operator+(difference_type i) const { return Iterator{m_blocks_begin, m_index + i}; }
        constexpr Iterator& operator+=(difference_type i)
        {
            m_index += i;
            return *this;
        }
        constexpr Iterator        operator-(difference_type i) const { return *this + (-i); }
        constexpr Iterator&       operator-=(difference_type i) { return *this += (-i); }
        constexpr Iterator&       operator++() { return *this += 1; }
        constexpr Iterator        operator++(int) const { return *this + 1; }
        constexpr Iterator&       operator--() { return *this -= 1; }
        constexpr Iterator        operator--(int) const { return *this - 1; }
        constexpr difference_type operator-(Iterator other) const { return m_index - other.m_index; }

    private:
        ConstexprUniquePtr< block_t >* m_blocks_begin{};
        ptrdiff_t                      m_index{};
    };
    using ConstIterator        = ConstexprRefStableCollection< const T >::Iterator;
    using ReverseIterator      = std::reverse_iterator< Iterator >;
    using ConstReverseIterator = std::reverse_iterator< ConstIterator >;

    constexpr ConstexprRefStableCollection() = default;

    constexpr T& operator[](size_t pos)
    {
        const auto [block_ind, el_pos] = getInds(pos);
        return m_blocks[block_ind]->operator[](el_pos);
    }
    constexpr const T& operator[](size_t pos) const
    {
        return const_cast< ConstexprRefStableCollection* >(this)->operator[](pos);
    }
    constexpr T&       front() { return this->operator[](0); }
    constexpr const T& front() const { return this->operator[](0); }
    constexpr T&       back() { return this->operator[](m_size - 1); }
    constexpr const T& back() const { return this->operator[](m_size - 1); }

    constexpr Iterator             begin() { return {m_blocks.data(), 0}; }
    constexpr ConstIterator        begin() const { return {m_blocks.data(), 0}; }
    constexpr ConstIterator        cbegin() const { return {m_blocks.data(), 0}; }
    constexpr Iterator             end() { return {m_blocks.data(), static_cast< ptrdiff_t >(m_size)}; }
    constexpr ConstIterator        end() const { return {m_blocks.data(), static_cast< ptrdiff_t >(m_size)}; }
    constexpr ConstIterator        cend() const { return {m_blocks.data(), static_cast< ptrdiff_t >(m_size)}; }
    constexpr ReverseIterator      rbegin() { return std::make_reverse_iterator(end()); }
    constexpr ConstReverseIterator rbegin() const { return std::make_reverse_iterator(end()); }
    constexpr ConstReverseIterator crbegin() const { return std::make_reverse_iterator(cend()); }
    constexpr ReverseIterator      rend() { return std::make_reverse_iterator(begin()); }
    constexpr ConstReverseIterator rend() const { return std::make_reverse_iterator(begin()); }
    constexpr ConstReverseIterator crend() const { return std::make_reverse_iterator(cbegin()); }

    template < typename... Arg >
    constexpr T& emplace(Arg&&... args)
        requires std::constructible_from< T, Arg... >
    {
        const auto [block_ind, el_pos] = getInds(m_size);
        if (block_ind >= m_blocks.size())
            grow();
        m_blocks[block_ind]->operator[](el_pos) = T(std::forward< Arg >(args)...);
        ++m_size;
        return back();
    }
    constexpr void push(const T& v)
        requires std::copy_constructible< T >
    {
        emplace(v);
    }

    [[nodiscard]] constexpr size_t size() const { return m_size; }

private:
    constexpr void grow() { m_blocks.emplaceBack(constexprMakeUnique< block_t >()); }

    ConstexprVector< ConstexprUniquePtr< block_t > > m_blocks;
    size_t                                           m_size = 0;
};

namespace detail
{
template < typename T >
inline constexpr bool is_rsc = false;
template < typename T, size_t BS >
inline constexpr bool is_rsc< ConstexprRefStableCollection< T, BS > > = true;
template < typename T >
concept RSC_c = is_rsc< T >;
template < typename T >
concept RscIterator_c =
    requires {
        typename T::container_type;
        typename T::value_type;
        T::container_type::block_size;
    } and
    std::same_as<
        T,
        typename ConstexprRefStableCollection< typename T::value_type, T::container_type::block_size >::Iterator >;
} // namespace detail

constexpr auto operator+(ptrdiff_t n, detail::RscIterator_c auto it)
{
    return it + n;
}
} // namespace lstr::util
#endif // L3STER_CONSTEXPRREFSTABLECOLLECTION_HPP
