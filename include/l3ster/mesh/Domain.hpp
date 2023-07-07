#ifndef L3STER_MESH_DOMAIN_HPP
#define L3STER_MESH_DOMAIN_HPP

#include "l3ster/defs/Enums.hpp"
#include "l3ster/mesh/Element.hpp"
#include "l3ster/mesh/ElementMeta.hpp"
#include "l3ster/util/Common.hpp"
#include "l3ster/util/TbbUtils.hpp"

#include <algorithm>
#include <execution>
#include <functional>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace lstr
{
struct SerializedDomain; // TODO
}

namespace lstr::mesh
{
template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
class Domain
{
    using Constraint = detail::ElementDeductionHelper< orders... >;

public:
    template < el_o_t... friend_orders >
        requires(sizeof...(friend_orders) > 0)
    friend class Domain;
    friend struct ::lstr::SerializedDomain;

    template < ElementType ET, el_o_t EO >
    using element_vector_t = std::vector< Element< ET, EO > >;
    using element_vector_variant_t =
        parametrize_type_over_element_types_and_orders_t< std::variant, element_vector_t, orders... >;
    using element_vector_variant_vector_t = std::vector< element_vector_variant_t >;
    using find_result_t                   = std::optional< element_ptr_variant_t< orders... > >;
    using const_find_result_t             = std::optional< element_cptr_variant_t< orders... > >;

    Domain() = default;
    Domain(element_vector_variant_vector_t&& element_vectors_, dim_t dim_)
        : m_element_vectors{std::move(element_vectors_)}, m_dim{dim_}
    {}

    template < ElementType ET, el_o_t EO >
    void push(const Element< ET, EO >& element);
    template < ElementType ET, el_o_t EO, typename... Args >
    void emplaceBack(Args&&... args);
    template < ElementType ET, el_o_t EO >
    auto getBackInserter();
    template < ElementType ET, el_o_t EO >
    void reserve(size_t size);

    template < typename F, SimpleExecutionPolicy_c ExecPolicy >
    void visit(F&& element_visitor, ExecPolicy&& policy)
        requires(Constraint::template invocable_on_elements< F >);
    template < typename F, SimpleExecutionPolicy_c ExecPolicy >
    void visit(F&& element_visitor, ExecPolicy&& policy) const
        requires(Constraint::template invocable_on_const_elements< F >);

    // Note: `zero` must be the identity element for the reduction
    auto
    transformReduce(const auto& zero, auto&& reduction, auto&& transform, SimpleExecutionPolicy_c auto&& policy) const
        -> std::decay_t< decltype(zero) >
        requires(Constraint::template invocable_on_const_elements_return< std::decay_t< decltype(zero) >,
                                                                          decltype(transform) > and
                 requires {
                     {
                         std::invoke(reduction, zero, zero)
                     } -> std::convertible_to< std::decay_t< decltype(zero) > >;
                 });

    template < typename F >
    auto find(F&& predicate, SimpleExecutionPolicy_c auto&& policy) -> find_result_t
        requires(Constraint::template invocable_on_const_elements_return< bool, F >);
    template < typename F >
    auto find(F&& predicate, SimpleExecutionPolicy_c auto&& policy) const -> const_find_result_t
        requires(Constraint::template invocable_on_const_elements_return< bool, F >);
    inline auto find(el_id_t id) -> find_result_t;
    inline auto find(el_id_t id) const -> const_find_result_t;

    [[nodiscard]] dim_t         getDim() const { return m_dim; };
    [[nodiscard]] inline size_t getNElements() const;

    template < el_o_t O_CONV >
    auto getConversionAlloc() const -> Domain< O_CONV >;

    template < ElementType ET, el_o_t EO >
    auto getElementVector() -> std::vector< Element< ET, EO > >&;

private:
    inline static auto constifyFound(const find_result_t& f) -> const_find_result_t
    {
        if (not f)
            return {};
        return util::constifyVariant(*f);
    }

    template < Access access >
    static auto wrapElementVisitor(auto&& element_visitor, SimpleExecutionPolicy_c auto&& policy);

    element_vector_variant_vector_t m_element_vectors;
    dim_t                           m_dim = 0;
};

namespace detail
{}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < ElementType ET, el_o_t EO >
auto Domain< orders... >::getElementVector() -> std::vector< Element< ET, EO > >&
{
    using el_vec_t = element_vector_t< ET, EO >;
    if (not m_element_vectors.empty())
        util::throwingAssert< std::invalid_argument >(Element< ET, EO >::native_dim == m_dim,
                                                      "Element dimension incompatible with domain dimension");
    else
        m_dim = Element< ET, EO >::native_dim;

    const auto vector_variant_it = std::find_if(m_element_vectors.begin(), m_element_vectors.end(), [](const auto& v) {
        return std::holds_alternative< el_vec_t >(v);
    });
    if (vector_variant_it == m_element_vectors.end())
        return std::get< el_vec_t >(m_element_vectors.emplace_back(std::in_place_type< el_vec_t >));
    else
        return std::get< el_vec_t >(*vector_variant_it);
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < ElementType ET, el_o_t EO >
void Domain< orders... >::push(const Element< ET, EO >& element)
{
    emplaceBack< ET, EO >(element);
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < ElementType T, el_o_t O, typename... ArgTypes >
void Domain< orders... >::emplaceBack(ArgTypes&&... Args)
{
    using el_t   = Element< T, O >;
    auto& vector = getElementVector< T, O >();
    vector.emplace_back(std::forward< ArgTypes >(Args)...);
    if (vector.size() == 1 or vector.back().getId() > (vector.crbegin() + 1)->getId()) [[likely]]
        return;
    else [[unlikely]]
        std::inplace_merge(vector.begin(), vector.end() - 1, vector.end(), [](const el_t& el1, const el_t& el2) {
            return el1.getId() < el2.getId();
        });
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < ElementType ET, el_o_t EO >
auto Domain< orders... >::getBackInserter()
{
    return std::back_inserter(getElementVector< ET, EO >());
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < ElementType ET, el_o_t EO >
void Domain< orders... >::reserve(size_t size)
{
    using el_vec_t = element_vector_t< ET, EO >;

    if (not m_element_vectors.empty())
        util::throwingAssert< std::invalid_argument >(Element< ET, EO >::native_dim == m_dim,
                                                      "Pushing element to domain of different dimension");
    else
        m_dim = Element< ET, EO >::native_dim;

    const auto vector_variant_it =
        std::ranges::find_if(m_element_vectors, [](const auto& v) { return std::holds_alternative< el_vec_t >(v); });
    if (vector_variant_it == m_element_vectors.end())
        std::get< el_vec_t >(m_element_vectors.emplace_back(std::in_place_type< el_vec_t >)).reserve(size);
    else
        std::get< el_vec_t >(*vector_variant_it).reserve(size);
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < typename F, SimpleExecutionPolicy_c ExecPolicy >
void Domain< orders... >::visit(F&& element_visitor, ExecPolicy&& policy)
    requires(Constraint::template invocable_on_elements< F >)
{
    const auto vec_variant_visitor = wrapElementVisitor< Access::ReadWrite >(element_visitor, policy);
    const auto invoke_visitor      = [&vec_variant_visitor](element_vector_variant_t& el_vec) {
        std::visit(vec_variant_visitor, el_vec);
    };
    if constexpr (SequencedPolicy_c< ExecPolicy >)
        std::ranges::for_each(m_element_vectors, invoke_visitor);
    else
        util::tbb::parallelFor(m_element_vectors, invoke_visitor);
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < typename F, SimpleExecutionPolicy_c ExecPolicy >
void Domain< orders... >::visit(F&& element_visitor, ExecPolicy&& policy) const
    requires(Constraint::template invocable_on_const_elements< F >)
{
    const auto vec_variant_visitor = wrapElementVisitor< Access::ReadOnly >(element_visitor, policy);
    const auto invoke_visitor      = [&vec_variant_visitor](const element_vector_variant_t& el_vec) {
        std::visit(vec_variant_visitor, el_vec);
    };
    if constexpr (SequencedPolicy_c< ExecPolicy >)
        std::ranges::for_each(m_element_vectors, invoke_visitor);
    else
        util::tbb::parallelFor(m_element_vectors, invoke_visitor);
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
auto Domain< orders... >::transformReduce(const auto&                    zero,
                                          auto&&                         reduction,
                                          auto&&                         transform,
                                          SimpleExecutionPolicy_c auto&& policy) const -> std::decay_t< decltype(zero) >
    requires(Constraint::template invocable_on_const_elements_return< std::decay_t< decltype(zero) >,
                                                                      decltype(transform) > and
             requires {
                 {
                     std::invoke(reduction, zero, zero)
                 } -> std::convertible_to< std::decay_t< decltype(zero) > >;
             })
{
    if constexpr (SequencedPolicy_c< decltype(policy) >)
    {
        const auto reduce_element_vector = [&]< ElementType ET, el_o_t EO >(const element_vector_t< ET, EO >& el_vec) {
            return std::transform_reduce(std::execution::unseq, begin(el_vec), end(el_vec), zero, reduction, transform);
        };
        const auto reduce_element_vector_variant = [&](const element_vector_variant_t& vec_var) {
            return std::visit< std::decay_t< decltype(zero) > >(reduce_element_vector, vec_var);
        };
        return std::transform_reduce(std::execution::unseq,
                                     begin(m_element_vectors),
                                     end(m_element_vectors),
                                     zero,
                                     reduction,
                                     reduce_element_vector_variant);
    }
    else
    {
        const auto reduce_element_vector = [&]< ElementType ET, el_o_t EO >(const element_vector_t< ET, EO >& el_vec) {
            return util::tbb::parallelTransformReduce(el_vec, zero, reduction, transform);
        };
        const auto reduce_element_vector_variant = [&](const element_vector_variant_t& vec_var) {
            return std::visit< std::decay_t< decltype(zero) > >(reduce_element_vector, vec_var);
        };
        return util::tbb::parallelTransformReduce(m_element_vectors, zero, reduction, reduce_element_vector_variant);
    }
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < typename F >
auto Domain< orders... >::find(F&& predicate, SimpleExecutionPolicy_c auto&& policy) -> find_result_t
    requires(Constraint::template invocable_on_const_elements_return< bool, F >)
{
    std::optional< element_ptr_variant_t< orders... > > opt_el_ptr_variant;
    std::mutex                                          mutex;
    const auto vector_visitor = [&]< ElementType T, el_o_t O >(const element_vector_t< T, O >& el_vec) {
        const auto el_it = std::find_if(policy, cbegin(el_vec), cend(el_vec), predicate);
        if (el_it != el_vec.cend())
        {
            const auto lock = std::lock_guard{mutex};
            opt_el_ptr_variant.emplace(const_cast< Element< T, O >* >(std::addressof(*el_it)));
            return true;
        }
        else
            return false;
    };
    const auto vector_variant_visitor = [&](const element_vector_variant_t& var) {
        return std::visit(vector_visitor, var);
    };
    std::find_if(policy, cbegin(m_element_vectors), cend(m_element_vectors), vector_variant_visitor);
    return opt_el_ptr_variant;
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < typename F >
auto Domain< orders... >::find(F&& predicate, SimpleExecutionPolicy_c auto&& policy) const -> const_find_result_t
    requires(Constraint::template invocable_on_const_elements_return< bool, F >)
{
    return constifyFound(const_cast< Domain* >(this)->find(predicate, std::forward< decltype(policy) >(policy)));
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
auto Domain< orders... >::find(el_id_t id) -> find_result_t
{
    std::optional< element_ptr_variant_t< orders... > > opt_el_ptr_variant;
    const auto vector_visitor = [&]< ElementType T, el_o_t O >(element_vector_t< T, O >& el_vec) -> bool {
        if (el_vec.empty())
            return false;

        const auto front_id = el_vec.front().getId();
        const auto back_id  = el_vec.back().getId();

        if (id < front_id or id > back_id)
            return false;

        // optimization for contiguous case
        if (back_id - front_id + 1u == el_vec.size())
        {
            opt_el_ptr_variant.emplace(std::addressof(el_vec[id - front_id]));
            return true;
        }

        const auto it = std::ranges::lower_bound(el_vec, id, {}, [](const Element< T, O >& el) { return el.getId(); });
        if (it == end(el_vec) or it->getId() != id)
            return false;

        opt_el_ptr_variant.emplace(std::addressof(*it));
        return true;
    };
    const auto vector_variant_visitor = [&](element_vector_variant_t& var) {
        return std::visit< bool >(vector_visitor, var);
    };
    std::ranges::find_if(m_element_vectors, vector_variant_visitor);
    return opt_el_ptr_variant;
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
auto Domain< orders... >::find(el_id_t id) const -> const_find_result_t
{
    return constifyFound(const_cast< Domain< orders... >* >(this)->find(id));
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < Access access >
auto Domain< orders... >::wrapElementVisitor(auto&& element_visitor, SimpleExecutionPolicy_c auto&& policy)
{
    if constexpr (access == Access::ReadWrite)
    {
        if constexpr (SequencedPolicy_c< decltype(policy) >)
            return [&element_visitor](auto& element_vector) {
                std::ranges::for_each(element_vector, element_visitor);
            };
        else
            return [&element_visitor](auto& element_vector) {
                util::tbb::parallelFor(element_vector, element_visitor);
            };
    }
    else
    {
        if constexpr (SequencedPolicy_c< decltype(policy) >)
            return [&element_visitor](const auto& element_vector) {
                std::ranges::for_each(element_vector, element_visitor);
            };
        else
            return [&element_visitor](const auto& element_vector) {
                util::tbb::parallelFor(element_vector, element_visitor);
            };
    }
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
size_t Domain< orders... >::getNElements() const
{
    return std::accumulate(
        begin(m_element_vectors), end(m_element_vectors), 0u, [](size_t sum, const auto& el_vec_var) {
            return sum + std::visit([](const auto& el_vec) { return el_vec.size(); }, el_vec_var);
        });
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
template < el_o_t O_CONV >
auto Domain< orders... >::getConversionAlloc() const -> Domain< O_CONV >
{
    Domain< O_CONV > retval;
    retval.m_element_vectors.reserve(m_element_vectors.size());
    retval.m_dim = m_dim;
    for (const auto& el_v : m_element_vectors)
    {
        std::visit(
            [&]< ElementType T, el_o_t O >(const element_vector_t< T, O >& existing_vec) {
                retval.template reserve< T, O_CONV >(existing_vec.size());
            },
            el_v);
    }
    return retval;
}

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
class DomainView
{
public:
    DomainView(const Domain< orders... >& domain_, d_id_t id_) : domain{std::addressof(domain_)}, id{id_} {}

    [[nodiscard]] d_id_t getID() const { return id; }
    [[nodiscard]] dim_t  getDim() const { return domain->getDim(); }
    [[nodiscard]] size_t getNElements() const { return domain->getNElements(); }

private:
    const Domain< orders... >* domain{};
    d_id_t                     id{};
};
} // namespace lstr::mesh
#endif // L3STER_MESH_DOMAIN_HPP
