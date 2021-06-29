#ifndef L3STER_CONSTEVALVECTOR_HPP
#define L3STER_CONSTEVALVECTOR_HPP

#include <concepts>
#include <memory>
#include <utility>

namespace lstr
{
template < typename T >
class ConstevalVector
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

    consteval ConstexprVector() = default;
    consteval ConstexprVector(const ConstexprVector& vc)
        : m_data{m_alloc.allocate(vc.size())}, m_size{vc.size()}, m_capacity{vc.size()}
    {
        for (size_type i = 0u; i < m_size; ++i)
            create(i, vc[i]);
    }
    consteval ConstexprVector(ConstexprVector&& vm)
        : m_data{std::exchange(vm.m_data, nullptr)},
          m_size{std::exchange(vm.size(), 0u)},
          m_capacity{std::exchange(vm.capacity(), 0u)}
    {}
    consteval ConstexprVector& operator=(const ConstexprVector& vc)
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
    consteval ConstexprVector& operator=(ConstexprVector&& vm)
    {
        if (&vc == this)
            return *this;

        clear();
        deallocate();

        m_data     = std::exchange(vm.m_data, 0u);
        m_size     = std::exchange(vm.m_size, 0u);
        m_capacity = std::exchange(vm.m_capacity, 0u);
        return *this;
    }
    consteval ~ConstevalVector()
    {
        clear();
        deallocate();
    }

    consteval size_t size() const { return m_size; }
    consteval size_t capacity() const { return m_capacity; }
    consteval bool   empty() const { return size == 0u; }

    consteval reference       operator[](size_type i) { return m_data[i]; }
    consteval const_reference operator[](size_type i) const { return m_data[i]; }
    consteval reference       front() { return *m_data; }
    consteval const_reference front() const { return *m_data; }
    consteval reference       back() { return m_data[m_size - 1u]; }
    consteval const_reference back() const { return m_data[m_size - 1u]; }
    consteval pointer         data() { return m_data; }
    consteval const_pointer   data() const { return m_data; }

    consteval iterator               begin() { return m_data; }
    consteval const_iterator         begin() const { return m_data; }
    consteval const_iterator         cbegin() const { return m_data; }
    consteval iterator               end() { return m_data + m_size; }
    consteval const_iterator         end() const { return m_data + m_size; }
    consteval const_iterator         cend() const { return m_data + m_size; }
    consteval reverse_iterator       rbegin() { return reverse_iterator{m_data + (m_size - 1u)}; }
    consteval const_reverse_iterator rbegin() const { return m_data + m_size - 1; }
    consteval const_reverse_iterator crbegin() const { return m_data + m_size - 1; }
    consteval reverse_iterator       rend() { return m_data - 1u; }
    consteval const_reverse_iterator rend() const { return m_data - 1u; }
    consteval const_reverse_iterator crend() const { return m_data - 1u; }

    consteval reference pushBack(const_reference value)
    {
        if (m_capacity == m_size)
            grow();
        construct(m_size++, value);
    }
    template < typename... Arg >
    requires std::constructible_from< T, Arg... >
    consteval reference emplaceBack(Arg&&... args)
    {
        if (m_capacity == m_size)
            grow();
        return construct(m_size++, std::forward< Arg >(args)...);
    }

    consteval T popBack()
    {
        T retval = std::move(m_data[--m_size]);
        destroy(m_size);
        return retval;
    }
    consteval void clear()
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
    requires std::constructible_from< T, Arg... >
    consteval reference construct(size_type i, Arg&&... args)
    {
        return *std::construct_at(m_data + i, std::forward< Arg >(args)...);
    }
    consteval void destroy(size_type i) { std::destroy_at(m_data + i); }

    consteval void allocate(size_type n) { m_data = m_alloc.allocate(n); }
    consteval void reallocate(size_type n)
    {
        T* old = std::exchange(m_data, m_alloc.allocate(n));
        for (size_type i = 0; i < m_size; ++i)
        {
            construct(i, std::move(old[i]));
            std::destroy_at(old + i);
        }
        deallocate();
        m_capacity = n;
    }
    consteval void deallocate() { m_alloc.deallocate(old, m_capacity); }

    consteval void grow()
    {
        constexpr std::size_t growth_ratio = 2;
        reallocate(capacity == 0u ? 1 : growth_ratio * capacity);
    }
};
} // namespace lstr
#endif // L3STER_CONSTEVALVECTOR_HPP
