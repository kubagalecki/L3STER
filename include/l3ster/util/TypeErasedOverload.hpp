#ifndef L3STER_UTIL_TYPEERASEDOVERLOAD_HPP
#define L3STER_UTIL_TYPEERASEDOVERLOAD_HPP

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>

namespace lstr::util
{
namespace detail
{
inline constexpr size_t type_erased_overload_sso_size = 2 * sizeof(void*);
static_assert(type_erased_overload_sso_size >= sizeof(void*));
template < typename Fun >
concept SsoCapableTypeErasedFun_c = sizeof(Fun) <= type_erased_overload_sso_size and alignof(Fun) <= alignof(void*);

template < typename Fun >
Fun* getTargetPtr(void* buf_ptr)
{
    if (SsoCapableTypeErasedFun_c< Fun >) // The target resides directly in the buffer
        return reinterpret_cast< Fun* >(buf_ptr);
    else // The buffer holds a void* to the target
    {
        const auto ptr_to_target = *reinterpret_cast< void** >(buf_ptr);
        return reinterpret_cast< Fun* >(ptr_to_target);
    }
}

template < typename Fun >
const Fun* getTargetPtr(const void* ptr)
{
    return getTargetPtr< Fun >(const_cast< void* >(ptr));
}

template < typename Return, typename Arg, typename CRTP >
class TypeErasedInvoker
{
public:
    TypeErasedInvoker() = default;
    template < typename Fun >
    TypeErasedInvoker(std::type_identity< Fun >)
        requires std::is_invocable_r_v< Return, std::add_const_t< Fun >, Arg >
        : m_invoker{+[](const void* fun_ptr, Arg arg) -> Return {
              const auto target_ptr = detail::getTargetPtr< std::add_const_t< Fun > >(fun_ptr);
              return std::invoke(*target_ptr, arg);
          }}
    {}

    auto operator()(Arg arg) const -> Return
    {
        const auto target_ptr = static_cast< const CRTP* >(this)->getTargetPtr();
        return std::invoke(m_invoker, target_ptr, arg);
    }

protected:
    void copyInvoker(const TypeErasedInvoker& other) { m_invoker = other.m_invoker; }

private:
    Return (*m_invoker)(const void*, Arg) = nullptr;
};
} // namespace detail

template < typename Return, typename... Args >
class TypeErasedOverload : public detail::TypeErasedInvoker< Return, Args, TypeErasedOverload< Return, Args... > >...
{
    // Due to lack of variadic friendship this is as fine-grained as we can get
    template < typename Ret, typename Arg, typename CRTP >
    friend class detail::TypeErasedInvoker;

    enum struct OperationType
    {
        Move,
        Destroy
    };

    template < typename Fun >
    static auto makeSpecialMemberImpl() -> void (*)(void*, void*, OperationType);
    template < typename Fun >
    static void initTarget(Fun&& f, std::byte* buf);

    inline void moveImpl(TypeErasedOverload&& other);
    inline void dtorImpl();
    inline void copyInvokersImpl(const TypeErasedOverload& other);
    auto        getTargetPtr() const -> const void* { return m_target_buf.data(); }

public:
    TypeErasedOverload() = default;
    template < typename Fun >
    TypeErasedOverload(Fun&& fun)
        requires((std::is_invocable_r_v< Return, const std::remove_cvref_t< Fun >&, Args > and ...) and
                 std::is_nothrow_move_constructible_v< std::remove_cvref_t< Fun > >);

    TypeErasedOverload(const TypeErasedOverload&)            = delete;
    TypeErasedOverload& operator=(const TypeErasedOverload&) = delete;
    TypeErasedOverload(TypeErasedOverload&& other) noexcept;
    TypeErasedOverload& operator=(TypeErasedOverload&& other) noexcept;
    ~TypeErasedOverload() { dtorImpl(); }

    explicit operator bool() const { return m_special_members; }
    using detail::TypeErasedInvoker< Return, Args, TypeErasedOverload< Return, Args... > >::operator()...;

private:
    alignas(void*) std::array< std::byte, detail::type_erased_overload_sso_size > m_target_buf;
    void (*m_special_members)(void*, void*, OperationType) = nullptr;
};

template < typename Return, typename... Args >
template < typename Fun >
TypeErasedOverload< Return, Args... >::TypeErasedOverload(Fun&& fun)
    requires((std::is_invocable_r_v< Return, const std::remove_cvref_t< Fun >&, Args > and ...) and
             std::is_nothrow_move_constructible_v< std::remove_cvref_t< Fun > >)
    : detail::TypeErasedInvoker< Return, Args, TypeErasedOverload< Return, Args... > >(
          std::type_identity< std::remove_cvref_t< Fun > >{})...,
      m_special_members{makeSpecialMemberImpl< std::remove_cvref_t< Fun > >()}
{
    initTarget(std::forward< Fun >(fun), m_target_buf.data());
}

template < typename Return, typename... Args >
TypeErasedOverload< Return, Args... >::TypeErasedOverload(TypeErasedOverload&& other) noexcept
{
    moveImpl(std::move(other));
}

template < typename Return, typename... Args >
TypeErasedOverload< Return, Args... >&
TypeErasedOverload< Return, Args... >::operator=(TypeErasedOverload&& other) noexcept
{
    if (this != &other)
    {
        dtorImpl();
        moveImpl(std::move(other));
    }
    return *this;
}

template < typename Return, typename... Args >
template < typename Fun >
void TypeErasedOverload< Return, Args... >::initTarget(Fun&& f, std::byte* buf)
{
    using target_t = std::remove_cvref_t< Fun >;
    if constexpr (detail::SsoCapableTypeErasedFun_c< target_t >)
        new (buf) target_t{std::forward< Fun >(f)};
    else
    {
        const void* alloc_ptr = new target_t{std::forward< Fun >(f)};
        std::memcpy(buf, &alloc_ptr, sizeof alloc_ptr);
    }
}

template < typename Return, typename... Args >
template < typename Fun >
auto TypeErasedOverload< Return, Args... >::makeSpecialMemberImpl() -> void (*)(void*, void*, OperationType)
{
    return +[](void* my_target_buf, void* other_target_buf, OperationType op) {
        const auto my_target = detail::getTargetPtr< Fun >(my_target_buf);
        switch (op)
        {
        case OperationType::Move:
            if constexpr (detail::SsoCapableTypeErasedFun_c< Fun >)
            {
                const auto other_target = detail::getTargetPtr< Fun >(other_target_buf);
                std::construct_at(my_target, std::move(*other_target));
            }
            else
            {
                std::memcpy(my_target_buf, other_target_buf, sizeof(void*));
                *reinterpret_cast< void** >(other_target_buf) = nullptr;
            }
            return;
        case OperationType::Destroy:
            if constexpr (detail::SsoCapableTypeErasedFun_c< Fun >)
                std::destroy_at(my_target);
            else
                delete my_target;
            return;
        }
    };
}

template < typename Return, typename... Args >
void TypeErasedOverload< Return, Args... >::moveImpl(TypeErasedOverload&& other)
{
    m_special_members = other.m_special_members;
    if (m_special_members)
    {
        copyInvokersImpl(other);
        std::invoke(m_special_members, m_target_buf.data(), other.m_target_buf.data(), OperationType::Move);
    }
}

template < typename Return, typename... Args >
void TypeErasedOverload< Return, Args... >::dtorImpl()
{
    if (m_special_members)
        std::invoke(m_special_members, m_target_buf.data(), nullptr, OperationType::Destroy);
}

template < typename Return, typename... Args >
void TypeErasedOverload< Return, Args... >::copyInvokersImpl(const TypeErasedOverload& other)
{
    (detail::TypeErasedInvoker< Return, Args, TypeErasedOverload< Return, Args... > >::copyInvoker(other), ...);
}
} // namespace lstr::util
#endif // L3STER_UTIL_TYPEERASEDOVERLOAD_HPP
