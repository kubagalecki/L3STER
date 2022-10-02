#ifndef L3STER_CONSTEXPRVECTOR_HPP
#define L3STER_CONSTEXPRVECTOR_HPP

#include <concepts>
#include <memory>
#include <utility>

namespace lstr
{
template < typename T >
class ConstexprVector
{
public:
    using value_type             = T;
    using allocator_type         = std::allocator< T >;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using reference              = T&;
    using const_reference        = const T&;
    using pointer                = T*;
    using const_pointer          = const T*;
    using iterator               = T*;
    using const_iterator         = const T*;
    using reverse_iterator       = std::reverse_iterator< iterator >;
    using const_reverse_iterator = std::reverse_iterator< const_iterator >;

    constexpr ConstexprVector() = default;
    constexpr ConstexprVector(size_type n, const T& value) : m_data{m_alloc.allocate(n)}, m_size{n}, m_capacity{n}
    {
        for (size_type i = 0u; i < n; ++i)
            construct(i, value);
    }
    constexpr ConstexprVector(size_type n) : m_data{m_alloc.allocate(n)}, m_size{n}, m_capacity{n}
    {
        for (size_type i = 0u; i < n; ++i)
            construct(i);
    }
    constexpr ConstexprVector(std::initializer_list< T > list)
        : m_data{m_alloc.allocate(list.size())}, m_size{list.size()}, m_capacity{list.size()}
    {
        auto it = list.begin();
        for (size_type i = 0u; i < m_size; ++i)
            construct(i, *it++);
    }

    constexpr ConstexprVector(const ConstexprVector& vc)
        : m_data{m_alloc.allocate(vc.size())}, m_size{vc.size()}, m_capacity{vc.size()}
    {
        for (size_type i = 0u; i < m_size; ++i)
            construct(i, vc[i]);
    }
    constexpr ConstexprVector(ConstexprVector&& vm) noexcept
        : m_data{std::exchange(vm.m_data, nullptr)},
          m_size{std::exchange(vm.m_size, 0u)},
          m_capacity{std::exchange(vm.m_capacity, 0u)}
    {}
    constexpr ConstexprVector& operator=(const ConstexprVector& vc)
    {
        if (&vc == this)
            return *this;

        if (vc.size() > m_capacity)
        {
            clear();
            reallocate(vc.size());
            for (size_type i = 0u; i < vc.size(); ++i)
                create(i, vc[i]);
        }
        else
        {
            size_type i = 0u;
            for (; i < m_size && i < vc.size(); ++i)
                m_data[i] = vc[i];
            for (; i < vc.size(); ++i)
                create(i, vc[i]);
            for (; i < m_size; ++i)
                destroy(i);
        }
        m_size = vc.size();
        return *this;
    }
    constexpr ConstexprVector& operator=(ConstexprVector&& vm) noexcept
    {
        if (&vm == this)
            return *this;

        clear();
        deallocate();

        m_data     = std::exchange(vm.m_data, nullptr);
        m_size     = std::exchange(vm.m_size, 0u);
        m_capacity = std::exchange(vm.m_capacity, 0u);
        return *this;
    }
    constexpr ~ConstexprVector()
    {
        clear();
        deallocate();
    }

    // Capacity
    [[nodiscard]] constexpr size_t size() const { return m_size; }
    [[nodiscard]] constexpr size_t capacity() const { return m_capacity; }
    [[nodiscard]] constexpr bool   empty() const { return m_size == 0u; }
    constexpr void                 reserve(size_type n)
    {
        if (n > m_capacity)
            reallocate(n);
    }

    // Iterators
    constexpr reference       operator[](size_type i) { return m_data[i]; }
    constexpr const_reference operator[](size_type i) const { return m_data[i]; }
    constexpr reference       front() { return *m_data; }
    constexpr const_reference front() const { return *m_data; }
    constexpr reference       back() { return m_data[m_size - 1u]; }
    constexpr const_reference back() const { return m_data[m_size - 1u]; }
    constexpr pointer         data() { return m_data; }
    constexpr const_pointer   data() const { return m_data; }

    constexpr iterator               begin() { return m_data; }
    constexpr const_iterator         begin() const { return m_data; }
    constexpr const_iterator         cbegin() const { return m_data; }
    constexpr iterator               end() { return m_data + m_size; }
    constexpr const_iterator         end() const { return m_data + m_size; }
    constexpr const_iterator         cend() const { return m_data + m_size; }
    constexpr reverse_iterator       rbegin() { return std::make_reverse_iterator(m_data + m_size); }
    constexpr const_reverse_iterator rbegin() const { return std::make_reverse_iterator(m_data + m_size); }
    constexpr const_reverse_iterator crbegin() const { return std::make_reverse_iterator(m_data + m_size); }
    constexpr reverse_iterator       rend() { return std::make_reverse_iterator(m_data); }
    constexpr const_reverse_iterator rend() const { return std::make_reverse_iterator(m_data); }
    constexpr const_reverse_iterator crend() const { return std::make_reverse_iterator(m_data); }

    constexpr void pushBack(const_reference value)
        requires std::copy_constructible< T >
    {
        emplaceBack(value);
    }
    template < typename... Arg >
    constexpr reference emplaceBack(Arg&&... args)
        requires std::constructible_from< T, Arg... >
    {
        if (m_capacity == m_size)
            grow();
        construct(m_size++, std::forward< Arg >(args)...);
        return back();
    }

    constexpr void popBack() { destroy(--m_size); }
    constexpr void clear()
    {
        for (size_type i = 0; i < m_size; ++i)
            destroy(i);
        m_size = 0u;
    }

private:
    allocator_type m_alloc{};
    pointer        m_data{nullptr};
    size_type      m_size{0}, m_capacity{0};

    template < typename... Arg >
    constexpr void construct(size_type i, Arg&&... args)
        requires std::constructible_from< T, Arg... >
    {
        std::construct_at(m_data + i, std::forward< Arg >(args)...);
    }
    constexpr void destroy(size_type i) { std::destroy_at(m_data + i); }

    constexpr void reallocate(size_type n)
    {
        T* old = std::exchange(m_data, m_alloc.allocate(n));
        for (size_type i = 0; i < m_size; ++i)
        {
            construct(i, std::move(old[i])); // NOLINT
            std::destroy_at(old + i);
        }
        if (old)
            m_alloc.deallocate(old, m_capacity);
        m_capacity = n;
    }
    constexpr void deallocate()
    {
        if (m_data)
            m_alloc.deallocate(m_data, m_capacity);
    }

    constexpr void grow()
    {
        constexpr std::size_t growth_ratio = 2;
        reallocate(m_capacity == 0u ? 1u : growth_ratio * m_capacity);
    }
};
} // namespace lstr
#endif // L3STER_CONSTEXPRVECTOR_HPP
