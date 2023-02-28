#ifndef L3STER_MESH_DOMAIN_HPP
#define L3STER_MESH_DOMAIN_HPP

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
class Domain
{
public:
    template < ElementTypes ET, el_o_t EO >
    using element_vector_t         = std::vector< Element< ET, EO > >;
    using element_vector_variant_t = parametrize_type_over_element_types_and_orders_t< std::variant, element_vector_t >;
    using element_vector_variant_vector_t = std::vector< element_vector_variant_t >;
    using find_result_t                   = std::optional< element_ptr_variant_t >;
    using const_find_result_t             = std::optional< element_cptr_variant_t >;

    friend struct SerializedDomain;

    Domain() = default;
    Domain(element_vector_variant_vector_t&& element_vectors_, dim_t dim_)
        : m_element_vectors{std::move(element_vectors_)}, m_dim{dim_}
    {}

    template < ElementTypes ET, el_o_t EO >
    void push(const Element< ET, EO >& element);
    template < ElementTypes ET, el_o_t EO, typename... Args >
    void emplaceBack(Args&&... args);
    template < ElementTypes ET, el_o_t EO >
    auto getBackInserter();
    template < ElementTypes ET, el_o_t EO >
    void reserve(size_t size);

    template < SimpleExecutionPolicy_c ExecPolicy >
    void visit(invocable_on_elements auto&& element_visitor, ExecPolicy&& policy);
    template < SimpleExecutionPolicy_c ExecPolicy >
    void visit(invocable_on_const_elements auto&& element_visitor, ExecPolicy&& policy) const;

    // Note: `zero` must be the identity element for the reduction
    auto
    transformReduce(const auto& zero, auto&& reduction, auto&& transform, SimpleExecutionPolicy_c auto&& policy) const
        -> std::decay_t< decltype(zero) >
        requires invocable_on_const_elements_r< decltype(transform), std::decay_t< decltype(zero) > > and
                 requires {
                     {
                         std::invoke(reduction, zero, zero)
                     } -> std::convertible_to< std::decay_t< decltype(zero) > >;
                 };

    [[nodiscard]] find_result_t              find(invocable_on_const_elements_r< bool > auto&& predicate,
                                                  SimpleExecutionPolicy_c auto&&               policy);
    [[nodiscard]] const_find_result_t        find(invocable_on_const_elements_r< bool > auto&& predicate,
                                                  SimpleExecutionPolicy_c auto&&               policy) const;
    [[nodiscard]] inline find_result_t       find(el_id_t id);
    [[nodiscard]] inline const_find_result_t find(el_id_t id) const;

    [[nodiscard]] dim_t         getDim() const { return m_dim; };
    [[nodiscard]] inline size_t getNElements() const;

    template < el_o_t O_CONV >
    [[nodiscard]] Domain getConversionAlloc() const;

    template < ElementTypes ET, el_o_t EO >
    std::vector< Element< ET, EO > >& getElementVector();

private:
    template < Access access >
    static auto wrapElementVisitor(auto&& element_visitor, SimpleExecutionPolicy_c auto&& policy);

    element_vector_variant_vector_t m_element_vectors;
    dim_t                           m_dim = 0;
};

namespace detail
{
inline Domain::const_find_result_t constifyFound(const Domain::find_result_t& f)
{
    if (not f)
        return {};
    return constifyVariant(*f);
}
} // namespace detail

template < ElementTypes ET, el_o_t EO >
std::vector< Element< ET, EO > >& Domain::getElementVector()
{
    using el_vec_t = element_vector_t< ET, EO >;
    if (not m_element_vectors.empty())
    {
        if (Element< ET, EO >::native_dim != m_dim)
            throw std::invalid_argument("Element dimension incompatible with domain dimension");
    }
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

template < ElementTypes ET, el_o_t EO >
void Domain::push(const Element< ET, EO >& element)
{
    emplaceBack< ET, EO >(element);
}

template < ElementTypes T, el_o_t O, typename... ArgTypes >
void Domain::emplaceBack(ArgTypes&&... Args)
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

template < ElementTypes ET, el_o_t EO >
auto Domain::getBackInserter()
{
    return std::back_inserter(getElementVector< ET, EO >());
}

template < ElementTypes ET, el_o_t EO >
void Domain::reserve(size_t size)
{
    using el_vec_t = element_vector_t< ET, EO >;

    if (not m_element_vectors.empty())
    {
        if (Element< ET, EO >::native_dim != m_dim)
            throw std::invalid_argument("Pushing element to domain of different dimension");
    }
    else
        m_dim = Element< ET, EO >::native_dim;

    const auto vector_variant_it =
        std::ranges::find_if(m_element_vectors, [](const auto& v) { return std::holds_alternative< el_vec_t >(v); });
    if (vector_variant_it == m_element_vectors.end())
        std::get< el_vec_t >(m_element_vectors.emplace_back(std::in_place_type< el_vec_t >)).reserve(size);
    else
        std::get< el_vec_t >(*vector_variant_it).reserve(size);
}

template < SimpleExecutionPolicy_c ExecPolicy >
void Domain::visit(invocable_on_elements auto&& element_visitor, ExecPolicy&& policy)
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

template < SimpleExecutionPolicy_c ExecPolicy >
void Domain::visit(invocable_on_const_elements auto&& element_visitor, ExecPolicy&& policy) const
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

auto Domain::transformReduce(const auto&                    zero,
                             auto&&                         reduction,
                             auto&&                         transform,
                             SimpleExecutionPolicy_c auto&& policy) const -> std::decay_t< decltype(zero) >
    requires invocable_on_const_elements_r< decltype(transform), std::decay_t< decltype(zero) > > and
             requires {
                 {
                     std::invoke(reduction, zero, zero)
                 } -> std::convertible_to< std::decay_t< decltype(zero) > >;
             }
{
    if constexpr (SequencedPolicy_c< decltype(policy) >)
    {
        const auto reduce_element_vector = [&]< ElementTypes ET, el_o_t EO >(const element_vector_t< ET, EO >& el_vec) {
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
        const auto reduce_element_vector = [&]< ElementTypes ET, el_o_t EO >(const element_vector_t< ET, EO >& el_vec) {
            return util::tbb::parallelTransformReduce(el_vec, zero, reduction, transform);
        };
        const auto reduce_element_vector_variant = [&](const element_vector_variant_t& vec_var) {
            return std::visit< std::decay_t< decltype(zero) > >(reduce_element_vector, vec_var);
        };
        return util::tbb::parallelTransformReduce(m_element_vectors, zero, reduction, reduce_element_vector_variant);
    }
}

Domain::find_result_t Domain::find(invocable_on_const_elements_r< bool > auto&& predicate,
                                   SimpleExecutionPolicy_c auto&&               policy)
{
    std::optional< element_ptr_variant_t > opt_el_ptr_variant;
    std::mutex                             mut;
    const auto vector_visitor = [&]< ElementTypes T, el_o_t O >(const element_vector_t< T, O >& el_vec) {
        const auto el_it = std::find_if(policy, cbegin(el_vec), cend(el_vec), predicate);
        if (el_it != el_vec.cend())
        {
            std::lock_guard lock{mut};
            opt_el_ptr_variant.emplace(const_cast< Element< T, O >* >(&*el_it));
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

Domain::const_find_result_t Domain::find(invocable_on_const_elements_r< bool > auto&& predicate,
                                         SimpleExecutionPolicy_c auto&&               policy) const
{
    return detail::constifyFound(
        const_cast< Domain* >(this)->find(predicate, std::forward< decltype(policy) >(policy)));
}

Domain::find_result_t Domain::find(el_id_t id)
{
    std::optional< element_ptr_variant_t > opt_el_ptr_variant;
    const auto vector_visitor = [&]< ElementTypes T, el_o_t O >(element_vector_t< T, O >& el_vec) -> bool {
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

Domain::const_find_result_t Domain::find(el_id_t id) const
{
    return detail::constifyFound(const_cast< Domain* >(this)->find(id));
}

template < Access access >
auto Domain::wrapElementVisitor(auto&& element_visitor, SimpleExecutionPolicy_c auto&& policy)
{
    if constexpr (access == Access::ReadWrite)
    {
        if constexpr (SequencedPolicy_c< decltype(policy) >)
            return [&element_visitor](auto& element_vector) {
                std::ranges::for_each(element_vector, element_visitor);
            };
        else
            return [&element_visitor, &policy](auto& element_vector) {
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
            return [&element_visitor, &policy](const auto& element_vector) {
                util::tbb::parallelFor(element_vector, element_visitor);
            };
    }
}

size_t Domain::getNElements() const
{
    return std::accumulate(
        begin(m_element_vectors), end(m_element_vectors), 0u, [](size_t sum, const auto& el_vec_var) {
            return sum + std::visit([](const auto& el_vec) { return el_vec.size(); }, el_vec_var);
        });
}

template < el_o_t O_CONV >
Domain Domain::getConversionAlloc() const
{
    Domain alloc;
    alloc.m_element_vectors.reserve(m_element_vectors.size());
    alloc.m_dim = m_dim;
    for (const auto& el_v : m_element_vectors)
    {
        std::visit(
            [&]< ElementTypes T, el_o_t O >(const element_vector_t< T, O >& existing_vec) {
                alloc.reserve< T, O_CONV >(existing_vec.size());
            },
            el_v);
    }
    return alloc;
}

class DomainView
{
public:
    DomainView() = default;
    DomainView(const Domain& domain_, d_id_t id_) : domain{std::addressof(domain_)}, id{id_} {}

    [[nodiscard]] d_id_t getID() const { return id; }
    [[nodiscard]] dim_t  getDim() const { return domain->getDim(); }
    [[nodiscard]] size_t getNElements() const { return domain->getNElements(); }

private:
    const Domain* domain = nullptr;
    d_id_t        id;
};
} // namespace lstr
#endif // L3STER_MESH_DOMAIN_HPP
