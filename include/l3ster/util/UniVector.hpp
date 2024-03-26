#ifndef L3STER_UTIL_UNIVECTOR_HPP
#define L3STER_UTIL_UNIVECTOR_HPP

#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/TbbUtils.hpp"

#include <tuple>
#include <variant>
#include <vector>

namespace lstr::util
{
/// Abstraction for multiple std::vectors of different types sharing an iteration space
template < typename... Ts >
class UniVector
{
    static_assert((std::is_object_v< Ts > and ...));
    static_assert(sizeof...(Ts) > 1);

public:
    using ptr_variant_t       = std::variant< std::add_pointer_t< Ts >... >;
    using const_ptr_variant_t = std::variant< std::add_pointer_t< std::add_const_t< Ts > >... >;

    static constexpr std::size_t num_types = sizeof...(Ts);

    template < typename T >
    auto getVector() -> std::vector< T >&;
    template < typename T >
    auto getVector() const -> const std::vector< T >&;

    [[nodiscard]] inline std::size_t size() const;
    [[nodiscard]] inline auto        sizes() const -> std::array< std::size_t, num_types >;
    [[nodiscard]] inline bool        empty() const;

    inline void reserve(const std::array< std::size_t, num_types >& sizes);

    template < typename Fun, SimpleExecutionPolicy_c ExecPolicy >
    void visit(Fun&& fun, ExecPolicy&&)
        requires(std::invocable< Fun, std::add_lvalue_reference_t< Ts > > and ...);
    template < typename Fun, SimpleExecutionPolicy_c ExecPolicy >
    void visit(Fun&& fun, ExecPolicy&&) const
        requires(std::invocable< Fun, std::add_lvalue_reference_t< std::add_const_t< Ts > > > and ...);
    template < typename Zero, typename Transform, typename Reduction, SimpleExecutionPolicy_c ExecPolicy >
    auto transformReduce(Zero zero, Transform transform, Reduction reduction, ExecPolicy&&) const -> Zero
        requires ReductionFor_c< Reduction, Zero > and (Mapping_c< Transform, Ts, Zero > and ...);
    template < typename Predicate >
    auto find(Predicate&& pred) const -> std::optional< const_ptr_variant_t >
        requires(std::predicate< Predicate, std::add_const_t< Ts > > and ...);
    template < typename Predicate >
    auto find(Predicate&& pred) -> std::optional< ptr_variant_t >
        requires(std::predicate< Predicate, std::add_const_t< Ts > > and ...);

    auto at(std::size_t index) -> ptr_variant_t;
    auto at(std::size_t index) const -> const_ptr_variant_t;

    template < typename VecVisitor >
    void visitVectors(VecVisitor&& v)
    {
        (std::invoke(v, getVector< Ts >()), ...);
    }
    template < typename VecVisitor >
    void visitVectors(VecVisitor&& v) const
    {
        (std::invoke(v, getVector< Ts >()), ...);
    }
    template < typename VecVisitor >
    void visitVectorsUntil(VecVisitor&& v)
    {
        (std::invoke(v, getVector< Ts >()) or ...);
    }
    template < typename VecVisitor >
    void visitVectorsUntil(VecVisitor&& v) const
    {
        (std::invoke(v, getVector< Ts >()) or ...);
    }

private:
    inline static auto deconstify(const_ptr_variant_t) -> ptr_variant_t;

    std::tuple< std::vector< Ts >... > m_contents;
};

template < typename... Ts >
template < typename Fun, SimpleExecutionPolicy_c ExecPolicy >
void UniVector< Ts... >::visit(Fun&& fun, ExecPolicy&&)
    requires(std::invocable< Fun, std::add_lvalue_reference_t< Ts > > and ...)
{
    if constexpr (std::same_as< std::execution::sequenced_policy, std::remove_cvref_t< ExecPolicy > >)
        visitVectors([&](auto& v) { std::ranges::for_each(v, fun); });
    else
    {
        oneapi::tbb::parallel_invoke([&] {
            util::tbb::parallelFor(getVector< Ts >(), fun);
        }...);
    }
}

template < typename... Ts >
template < typename Fun, SimpleExecutionPolicy_c ExecPolicy >
void UniVector< Ts... >::visit(Fun&& fun, ExecPolicy&&) const
    requires(std::invocable< Fun, std::add_lvalue_reference_t< std::add_const_t< Ts > > > and ...)
{
    if constexpr (std::same_as< std::execution::sequenced_policy, std::remove_cvref_t< ExecPolicy > >)
        visitVectors([&](const auto& v) { std::ranges::for_each(v, fun); });
    else
    {
        oneapi::tbb::parallel_invoke([&] {
            util::tbb::parallelFor(getVector< Ts >(), fun);
        }...);
    }
}

template < typename... Ts >
template < typename Zero, typename Transform, typename Reduction, SimpleExecutionPolicy_c ExecPolicy >
auto UniVector< Ts... >::transformReduce(Zero zero, Transform transform, Reduction reduction, ExecPolicy&&) const
    -> Zero
    requires ReductionFor_c< Reduction, Zero > and (Mapping_c< Transform, Ts, Zero > and ...)
{
    auto intermediate_reductions = std::array< Zero, sizeof...(Ts) >{};
    if constexpr (std::same_as< std::execution::sequenced_policy, std::remove_cvref_t< ExecPolicy > >)
    {
        const auto reduce_vec = [&]< std::size_t I >(std::integral_constant< std::size_t, I >) {
            const auto& vec            = std::get< I >(m_contents);
            intermediate_reductions[I] = std::transform_reduce(vec.begin(), vec.end(), zero, reduction, transform);
        };
        const auto deduct_helper = [&reduce_vec]< std::size_t... Is >(std::index_sequence< Is... >) {
            (reduce_vec(std::integral_constant< std::size_t, Is >{}), ...);
        };
        std::invoke(deduct_helper, std::make_index_sequence< sizeof...(Ts) >{});
    }
    else
    {
        const auto deduct_helper = [&]< std::size_t... Is >(std::index_sequence< Is... >) {
            oneapi::tbb::parallel_invoke([&] {
                constexpr auto index           = Is;
                const auto&    vec             = std::get< index >(m_contents);
                intermediate_reductions[index] = tbb::parallelTransformReduce(vec, zero, reduction, transform);
            }...);
        };
        std::invoke(deduct_helper, std::make_index_sequence< sizeof...(Ts) >{});
    }
    return std::reduce(intermediate_reductions.begin(), intermediate_reductions.end(), zero, reduction);
}

template < typename... Ts >
template < typename Predicate >
auto UniVector< Ts... >::find(Predicate&& pred) const -> std::optional< const_ptr_variant_t >
    requires(std::predicate< Predicate, std::add_const_t< Ts > > and ...)
{
    auto       retval      = std::optional< const_ptr_variant_t >{};
    const auto try_find_in = [&]< typename T >(const std::vector< T >& vec) {
        using cptr_t     = std::add_pointer_t< std::add_const_t< T > >;
        const auto iter  = std::ranges::find_if(vec, pred);
        const bool found = iter != vec.end();
        if (found)
            retval.emplace(std::in_place_type< cptr_t >, std::addressof(*iter));
        return found;
    };
    visitVectorsUntil(try_find_in);
    return retval;
}

template < typename... Ts >
template < typename Predicate >
auto UniVector< Ts... >::find(Predicate&& pred) -> std::optional< ptr_variant_t >
    requires(std::predicate< Predicate, std::add_const_t< Ts > > and ...)
{
    const auto found = std::as_const(*this).find(std::forward< Predicate >(pred));
    if (found)
        return deconstify(*found);
    else
        return {};
}

template < typename... Ts >
std::size_t UniVector< Ts... >::size() const
{
    return (getVector< Ts >().size() + ...);
}

template < typename... Ts >
auto UniVector< Ts... >::sizes() const -> std::array< std::size_t, num_types >
{
    return {getVector< Ts >().size()...};
}

template < typename... Ts >
bool UniVector< Ts... >::empty() const
{
    return ((getVector< Ts >().size() == 0) and ...);
}

template < typename... Ts >
void UniVector< Ts... >::reserve(const std::array< std::size_t, num_types >& sizes)
{
    auto reserve_for = [&sizes, i = 0](auto& vector) mutable {
        vector.reserve(sizes[i++]);
    };
    (reserve_for(getVector< Ts >()), ...);
}

template < typename... Ts >
auto UniVector< Ts... >::at(std::size_t index) const -> const_ptr_variant_t
{
    auto retval       = const_ptr_variant_t{};
    auto try_find_for = [&, i = size_t{}]< typename T >(std::type_identity< T >) mutable {
        const auto& vec = getVector< T >();
        if (index < i + vec.size())
        {
            retval = std::next(vec.data(), index - i);
            return true;
        }
        i += vec.size();
        return false;
    };
    const bool found = (try_find_for(std::type_identity< Ts >{}) or ...);
    throwingAssert< std::out_of_range >(found, "Out of bounds indexed access");
    return retval;
}

template < typename... Ts >
auto UniVector< Ts... >::at(std::size_t index) -> ptr_variant_t
{
    return deconstify(std::as_const(*this).at(index));
}

template < typename... Ts >
template < typename T >
auto UniVector< Ts... >::getVector() -> std::vector< T >&
{
    return std::get< std::vector< T > >(m_contents);
}

template < typename... Ts >
template < typename T >
auto UniVector< Ts... >::getVector() const -> const std::vector< T >&
{
    return std::get< std::vector< T > >(m_contents);
}

template < typename... Ts >
auto UniVector< Ts... >::deconstify(const_ptr_variant_t ptr_var) -> ptr_variant_t
{
    constexpr auto do_deconstify = []< typename T >(const T* ptr) -> ptr_variant_t {
        return const_cast< T* >(ptr);
    };
    return std::visit(do_deconstify, ptr_var);
}
} // namespace lstr::util
#endif // L3STER_UTIL_UNIVECTOR_HPP
