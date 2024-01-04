#ifndef L3STER_MESH_DOMAIN_HPP
#define L3STER_MESH_DOMAIN_HPP

#include "l3ster/common/Enums.hpp"
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

namespace lstr::mesh
{
template < typename F, el_o_t... orders >
concept MutableElementVisitor_c = ElementDeductionHelper< orders... >::template invocable_on_elements< F >;
template < typename F, el_o_t... orders >
concept ConstElementVisitor_c = ElementDeductionHelper< orders... >::template invocable_on_const_elements< F >;
template < typename F, el_o_t... orders >
concept ElementPredicate_c =
    ElementDeductionHelper< orders... >::template invocable_on_const_elements_return< bool, F >;
template < typename Zero, typename Transform, typename Reduction, el_o_t... orders >
concept TransformReducible_c =
    ElementDeductionHelper< orders... >::template invocable_on_const_elements_return< Zero, Transform > and
    ReductionFor_c< Reduction, Zero >;

namespace detail
{
template < el_o_t... orders >
auto makeMutableFindResult(std::optional< element_cptr_variant_t< orders... > > const_find)
    -> std::optional< element_ptr_variant_t< orders... > >
{
    auto retval = std::optional< element_ptr_variant_t< orders... > >{};
    if (const_find)
    {
        constexpr auto remove_const = []< ElementType ET, el_o_t EO >(const Element< ET, EO >* ptr) {
            return const_cast< Element< ET, EO >* >(ptr);
        };
        std::visit([&](auto ptr) { retval.emplace(remove_const(ptr)); }, *const_find);
    }
    return retval;
}
} // namespace detail

template < el_o_t... orders >
class Domain
{
    static_assert(sizeof...(orders) > 0);

public:
    template < el_o_t... friend_orders >
    friend class Domain;
    friend struct SerializedDomain;

    template < ElementType ET, el_o_t EO >
    using element_vector_t = std::vector< Element< ET, EO > >;
    using element_vector_variant_t =
        parametrize_type_over_element_types_and_orders_t< std::variant, element_vector_t, orders... >;
    using element_vector_variant_vector_t = std::vector< element_vector_variant_t >;
    using find_result_t                   = std::optional< element_ptr_variant_t< orders... > >;
    using const_find_result_t             = std::optional< element_cptr_variant_t< orders... > >;

    Domain() = default;
    Domain(element_vector_variant_vector_t element_vectors_, dim_t dim_)
        : m_element_vectors{std::move(element_vectors_)}, m_dim{dim_}
    {}

    template < ElementType ET, el_o_t EO >
    auto        getElementVector() -> std::vector< Element< ET, EO > >&;
    inline void sort();

    template < MutableElementVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy >
    void visit(Visitor&& element_visitor, ExecPolicy&& policy);
    template < ConstElementVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy >
    void visit(Visitor&& element_visitor, ExecPolicy&& policy) const;

    // Note: `zero` must be the identity element for the reduction
    template < typename Zero, typename Transform, typename Reduction, SimpleExecutionPolicy_c ExecPolicy >
    auto transformReduce(Zero zero, Transform transform, Reduction reduction, ExecPolicy&& policy) const -> Zero
        requires TransformReducible_c< Zero, Transform, Reduction, orders... >;

    template < ElementPredicate_c< orders... > Predicate, SimpleExecutionPolicy_c ExecPolicy >
    auto find(Predicate&& predicate, ExecPolicy&& policy) -> find_result_t;
    template < ElementPredicate_c< orders... > Predicate, SimpleExecutionPolicy_c ExecPolicy >
    auto        find(Predicate&& predicate, ExecPolicy&& policy) const -> const_find_result_t;
    inline auto find(el_id_t id) -> find_result_t;
    inline auto find(el_id_t id) const -> const_find_result_t;

    [[nodiscard]] dim_t         getDim() const { return m_dim; };
    [[nodiscard]] inline size_t getNElements() const;

    template < el_o_t O_CONV >
    auto getConversionAlloc() const -> Domain< O_CONV >;

private:
    // Deduce constness based on the elem vectors, helps with deduplication. Idea similar to C++23 "deducing this"
    template < typename ElVecs, typename Visitor, typename Policy >
    static void visitImpl(ElVecs&& element_vectors, Visitor&& visitor, Policy&& policy);

    element_vector_variant_vector_t m_element_vectors;
    dim_t                           m_dim = 0;
};

template < el_o_t... orders >
void Domain< orders... >::sort()
{
    constexpr auto sort_elvec = []< ElementType ET, el_o_t EO >(std::vector< Element< ET, EO > >& vec) {
        std::ranges::sort(vec, {}, &Element< ET, EO >::getId);
    };
    for (auto& el_vec : m_element_vectors)
        std::visit(sort_elvec, el_vec);
}

template < el_o_t... orders >
template < ElementType ET, el_o_t EO >
auto Domain< orders... >::getElementVector() -> std::vector< Element< ET, EO > >&
{
    using el_vec_t = element_vector_t< ET, EO >;
    if (not m_element_vectors.empty())
        util::throwingAssert< std::invalid_argument >(Element< ET, EO >::native_dim == m_dim,
                                                      "Element dimension differs from domain dimension");
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
template < MutableElementVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy >
void Domain< orders... >::visit(Visitor&& element_visitor, ExecPolicy&& policy)
{
    visitImpl(m_element_vectors, std::forward< Visitor >(element_visitor), std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < ConstElementVisitor_c< orders... > Visitor, SimpleExecutionPolicy_c ExecPolicy >
void Domain< orders... >::visit(Visitor&& element_visitor, ExecPolicy&& policy) const
{
    visitImpl(m_element_vectors, std::forward< Visitor >(element_visitor), std::forward< ExecPolicy >(policy));
}

template < el_o_t... orders >
template < typename Zero, typename Transform, typename Reduction, SimpleExecutionPolicy_c ExecPolicy >
auto Domain< orders... >::transformReduce(Zero zero, Transform trans, Reduction reduction, ExecPolicy&& policy) const
    -> Zero
    requires TransformReducible_c< Zero, Transform, Reduction, orders... >
{
    const auto reduce_el_vec = [&]< ElementType ET, el_o_t EO >(const element_vector_t< ET, EO >& el_vec) {
        return std::transform_reduce(policy, el_vec.begin(), el_vec.end(), zero, reduction, trans);
    };
    const auto trans_el_vec_var = [&](const element_vector_variant_t& vec_var) {
        return std::visit< std::decay_t< decltype(zero) > >(reduce_el_vec, vec_var);
    };
    return std::transform_reduce(
        policy, m_element_vectors.begin(), m_element_vectors.end(), zero, reduction, trans_el_vec_var);
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > Predicate, SimpleExecutionPolicy_c ExecPolicy >
auto Domain< orders... >::find(Predicate&& predicate, ExecPolicy&& policy) -> find_result_t
{
    const auto const_find =
        std::as_const(*this).find(std::forward< Predicate >(predicate), std::forward< ExecPolicy >(policy));
    return detail::makeMutableFindResult< orders... >(const_find);
}

template < el_o_t... orders >
template < ElementPredicate_c< orders... > Predicate, SimpleExecutionPolicy_c ExecPolicy >
auto Domain< orders... >::find(Predicate&& predicate, ExecPolicy&& policy) const -> const_find_result_t
{
    auto       retval         = const_find_result_t{};
    auto       mut            = std::mutex{};
    const auto vector_visitor = [&]< ElementType T, el_o_t O >(const element_vector_t< T, O >& el_vec) {
        const auto el_it = std::find_if(policy, cbegin(el_vec), cend(el_vec), predicate);
        if (el_it != el_vec.cend())
        {
            const auto lock = std::lock_guard{mut};
            retval.emplace(std::addressof(*el_it));
            return true;
        }
        else
            return false;
    };
    const auto vector_variant_visitor = [&](const element_vector_variant_t& var) {
        return std::visit(vector_visitor, var);
    };
    std::find_if(policy, cbegin(m_element_vectors), cend(m_element_vectors), vector_variant_visitor);
    return retval;
}

template < el_o_t... orders >
auto Domain< orders... >::find(el_id_t id) -> find_result_t
{
    const auto const_find = std::as_const(*this).find(id);
    return detail::makeMutableFindResult< orders... >(const_find);
}

template < el_o_t... orders >
auto Domain< orders... >::find(el_id_t id) const -> const_find_result_t
{
    auto       retval         = const_find_result_t{};
    const auto vector_visitor = [&]< ElementType T, el_o_t O >(const element_vector_t< T, O >& el_vec) -> bool {
        if (el_vec.empty())
            return false;

        const auto front_id = el_vec.front().getId();
        const auto back_id  = el_vec.back().getId();

        if (id < front_id or id > back_id)
            return false;

        // optimization for contiguous case
        if (back_id - front_id + 1u == el_vec.size())
        {
            retval.emplace(std::addressof(el_vec[id - front_id]));
            return true;
        }

        const auto it = std::ranges::lower_bound(el_vec, id, {}, [](const Element< T, O >& el) { return el.getId(); });
        if (it == end(el_vec) or it->getId() != id)
            return false;

        retval.emplace(std::addressof(*it));
        return true;
    };
    const auto vector_variant_visitor = [&](const element_vector_variant_t& var) {
        return std::visit< bool >(vector_visitor, var);
    };
    std::ranges::find_if(m_element_vectors, vector_variant_visitor);
    return retval;
}

template < el_o_t... orders >
template < typename ElVecs, typename Visitor, typename Policy >
void Domain< orders... >::visitImpl(ElVecs&& element_vectors, Visitor&& visitor, Policy&& policy)
{
    const auto visit_el_vec = [&](auto&& el_vec) {
        std::for_each(policy, el_vec.begin(), el_vec.end(), visitor);
    };
    const auto variant_dispatch = [&](auto&& variant) {
        std::visit(visit_el_vec, variant);
    };
    // std::for_each with sequential policy does not guarantee iteration order
    if constexpr (std::same_as< std::remove_cvref_t< Policy >, std::execution::sequenced_policy >)
        std::ranges::for_each(element_vectors, variant_dispatch);
    else
        std::for_each(policy, element_vectors.begin(), element_vectors.end(), variant_dispatch);
}

template < el_o_t... orders >
size_t Domain< orders... >::getNElements() const
{
    return std::accumulate(
        begin(m_element_vectors), end(m_element_vectors), 0u, [](size_t sum, const auto& el_vec_var) {
            return sum + std::visit([](const auto& el_vec) { return el_vec.size(); }, el_vec_var);
        });
}

template < el_o_t... orders >
template < el_o_t O_CONV >
auto Domain< orders... >::getConversionAlloc() const -> Domain< O_CONV >
{
    auto retval = Domain< O_CONV >{};
    retval.m_element_vectors.reserve(m_element_vectors.size());
    retval.m_dim = m_dim;
    for (const auto& el_vec : m_element_vectors)
    {
        const auto reserve = [&]< ElementType T, el_o_t O >(const element_vector_t< T, O >& existing_vec) {
            retval.template getElementVector< T, O_CONV >().reserve(existing_vec.size());
        };
        std::visit(reserve, el_vec);
    }
    return retval;
}

template < el_o_t... orders >
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
