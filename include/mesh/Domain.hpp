#ifndef L3STER_MESH_DOMAIN_HPP
#define L3STER_MESH_DOMAIN_HPP

#include "mesh/Aliases.hpp"
#include "mesh/Element.hpp"
#include "util/Concepts.hpp"

#include <algorithm>
#include <execution>
#include <functional>
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
    template < ElementTypes ELTYPE, el_o_t ELORDER >
    using element_vector_t         = std::vector< Element< ELTYPE, ELORDER > >;
    using element_vector_variant_t = parametrize_type_over_element_types_and_orders_t< std::variant, element_vector_t >;
    using element_vector_variant_vector_t = std::vector< element_vector_variant_t >;
    using find_result_t                   = std::optional< element_ptr_variant_t >;
    using const_find_result_t             = std::optional< element_cptr_variant_t >;

    friend struct SerializedDomain;

    Domain() = default;
    Domain(element_vector_variant_vector_t&& element_vectors_, dim_t dim_)
        : element_vectors{std::move(element_vectors_)}, dim{dim}
    {}

    template < ElementTypes ELTYPE, el_o_t ELORDER >
    void push(const Element< ELTYPE, ELORDER >& element);
    template < ElementTypes ELTYPE, el_o_t ELORDER, typename... Args >
    void emplaceBack(Args&&... args);
    template < ElementTypes ELTYPE, el_o_t ELORDER >
    auto getBackInserter();
    template < ElementTypes ELTYPE, el_o_t ELORDER >
    void reserve(size_t size);

    template < invocable_on_elements F, ExecutionPolicy_c ExecPolicy >
    void visit(F&& element_visitor, const ExecPolicy& policy);
    template < invocable_on_const_elements F, ExecutionPolicy_c ExecPolicy >
    void cvisit(F&& element_visitor, const ExecPolicy& policy) const;

    template < invocable_on_const_elements_r< bool > F, ExecutionPolicy_c ExecPolicy >
    [[nodiscard]] find_result_t find(F&& predicate, const ExecPolicy& policy);
    template < invocable_on_const_elements_r< bool > F, ExecutionPolicy_c ExecPolicy >
    [[nodiscard]] const_find_result_t        find(F&& predicate, const ExecPolicy& policy) const;
    [[nodiscard]] inline find_result_t       find(el_id_t id);
    [[nodiscard]] inline const_find_result_t find(el_id_t id) const;

    [[nodiscard]] dim_t         getDim() const { return dim; };
    [[nodiscard]] inline size_t getNElements() const;

    template < el_o_t O_CONV >
    [[nodiscard]] Domain getConversionAlloc() const;

    template < ElementTypes ELTYPE, el_o_t ELORDER >
    std::vector< Element< ELTYPE, ELORDER > >& getElementVector();

private:
    template < typename F, ExecutionPolicy_c ExecPolicy >
    static auto wrapElementVisitor(F& element_visitor, const ExecPolicy& policy);
    template < typename F, ExecutionPolicy_c ExecPolicy >
    static auto wrapCElementVisitor(F& element_visitor, const ExecPolicy& policy);

    element_vector_variant_vector_t element_vectors;
    dim_t                           dim = 0;
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

template < ElementTypes ELTYPE, el_o_t ELORDER >
std::vector< Element< ELTYPE, ELORDER > >& Domain::getElementVector()
{
    using el_vec_t = element_vector_t< ELTYPE, ELORDER >;
    if (not element_vectors.empty())
    {
        if (ElementTraits< Element< ELTYPE, ELORDER > >::native_dim != dim)
            throw std::invalid_argument("Element dimension incompatible with domain dimension");
    }
    else
        dim = ElementTraits< Element< ELTYPE, ELORDER > >::native_dim;

    const auto vector_variant_it = std::find_if(element_vectors.begin(), element_vectors.end(), [](const auto& v) {
        return std::holds_alternative< el_vec_t >(v);
    });
    if (vector_variant_it == element_vectors.end())
        return std::get< el_vec_t >(element_vectors.emplace_back(std::in_place_type< el_vec_t >));
    else
        return std::get< el_vec_t >(*vector_variant_it);
}

template < ElementTypes ELTYPE, el_o_t ELORDER >
void Domain::push(const Element< ELTYPE, ELORDER >& element)
{
    emplaceBack< ELTYPE, ELORDER >(element);
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

template < ElementTypes ELTYPE, el_o_t ELORDER >
auto Domain::getBackInserter()
{
    return std::back_inserter(getElementVector< ELTYPE, ELORDER >());
}

template < ElementTypes ELTYPE, el_o_t ELORDER >
void Domain::reserve(size_t size)
{
    using el_vec_t = element_vector_t< ELTYPE, ELORDER >;

    if (not element_vectors.empty())
    {
        if (ElementTraits< Element< ELTYPE, ELORDER > >::native_dim != dim)
            throw std::invalid_argument("Pushing element to domain of different dimension");
    }
    else
        dim = ElementTraits< Element< ELTYPE, ELORDER > >::native_dim;

    const auto vector_variant_it =
        std::ranges::find_if(element_vectors, [](const auto& v) { return std::holds_alternative< el_vec_t >(v); });
    if (vector_variant_it == element_vectors.end())
    {
        std::get< el_vec_t >(element_vectors.emplace_back(std::in_place_type< el_vec_t >)).reserve(size);
    }
    else
    {
        std::get< el_vec_t >(*vector_variant_it).reserve(size);
    }
}

template < invocable_on_elements F, ExecutionPolicy_c ExecPolicy >
void Domain::visit(F&& element_visitor, const ExecPolicy& policy)
{
    auto visitor = wrapElementVisitor(element_visitor, policy);
    std::ranges::for_each(element_vectors, [&](element_vector_variant_t& el_vec) { std::visit(visitor, el_vec); });
}

template < invocable_on_const_elements F, ExecutionPolicy_c ExecPolicy >
void Domain::cvisit(F&& element_visitor, const ExecPolicy& policy) const
{
    auto visitor = wrapCElementVisitor(element_visitor, policy);
    std::ranges::for_each(element_vectors,
                          [&](const element_vector_variant_t& el_vec) { std::visit(visitor, el_vec); });
}

template < invocable_on_const_elements_r< bool > F, ExecutionPolicy_c ExecPolicy >
Domain::find_result_t Domain::find(F&& predicate, const ExecPolicy& policy)
{
    std::optional< element_ptr_variant_t > opt_el_ptr_variant;
    const auto vector_visitor = [&]< ElementTypes T, el_o_t O >(const element_vector_t< T, O >& el_vec) {
        const auto el_it = std::find_if(policy, cbegin(el_vec), cend(el_vec), std::ref(predicate));
        if (el_it != el_vec.cend())
        {
            // const_cast is used because we don't want to allow the predicate to alter the elements
            opt_el_ptr_variant.emplace(const_cast< Element< T, O >* >(&*el_it));
            return true;
        }
        else
            return false;
    };
    const auto vector_variant_visitor = [&](const element_vector_variant_t& var) {
        return std::visit(vector_visitor, var);
    };
    std::ranges::find_if(element_vectors, vector_variant_visitor);
    return opt_el_ptr_variant;
}

template < invocable_on_const_elements_r< bool > F, ExecutionPolicy_c ExecPolicy >
Domain::const_find_result_t Domain::find(F&& predicate, const ExecPolicy& policy) const
{
    return detail::constifyFound(const_cast< Domain* >(this)->find(predicate, policy));
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
            opt_el_ptr_variant.emplace(&el_vec[id - front_id]);
            return true;
        }

        const auto it = std::ranges::lower_bound(el_vec, id, {}, [](const Element< T, O >& el) { return el.getId(); });
        if (it == end(el_vec) or it->getId() != id)
            return false;

        opt_el_ptr_variant.emplace(&*it);
        return true;
    };
    const auto vector_variant_visitor = [&](element_vector_variant_t& var) {
        return std::visit< bool >(vector_visitor, var);
    };
    std::ranges::find_if(element_vectors, vector_variant_visitor);
    return opt_el_ptr_variant;
}

Domain::const_find_result_t Domain::find(el_id_t id) const
{
    return detail::constifyFound(const_cast< Domain* >(this)->find(id));
}

template < typename F, ExecutionPolicy_c ExecPolicy >
auto Domain::wrapElementVisitor(F& element_visitor, const ExecPolicy& policy)
{
    return [&](auto& element_vector) {
        // we need to be able to iterate in a deterministic order, execution::seq does not guarantee that
        if constexpr (std::is_same_v< ExecPolicy, std::execution::sequenced_policy >)
            std::ranges::for_each(element_vector, std::ref(element_visitor));
        else
            std::for_each(policy, begin(element_vector), end(element_vector), std::ref(element_visitor));
    };
}

template < typename F, ExecutionPolicy_c ExecPolicy >
auto Domain::wrapCElementVisitor(F& element_visitor, const ExecPolicy& policy)
{
    return [&](const auto& element_vector) {
        // we need to be able to iterate in a deterministic order, execution::seq does not guarantee that
        if constexpr (std::is_same_v< ExecPolicy, std::execution::sequenced_policy >)
            std::ranges::for_each(element_vector, std::ref(element_visitor));
        else
            std::for_each(policy, cbegin(element_vector), cend(element_vector), std::ref(element_visitor));
    };
}

size_t Domain::getNElements() const
{
    return std::accumulate(
        element_vectors.cbegin(), element_vectors.cend(), 0u, [](size_t sum, const auto& el_vec_var) {
            return sum + std::visit([](const auto& el_vec) { return el_vec.size(); }, el_vec_var);
        });
}

template < el_o_t O_CONV >
Domain Domain::getConversionAlloc() const
{
    Domain alloc;
    alloc.element_vectors.reserve(element_vectors.size());
    alloc.dim = dim;
    for (const auto& el_v : element_vectors)
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
    using domain_ref_t = std::reference_wrapper< const Domain >;

public:
    DomainView(const Domain& domain_, d_id_t id_) : domain{std::cref(domain_)}, id{id_} {}

    [[nodiscard]] d_id_t getID() const { return id; }
    [[nodiscard]] dim_t  getDim() const { return domain.get().getDim(); }
    [[nodiscard]] size_t getNElements() const { return domain.get().getNElements(); }

    domain_ref_t domain;

private:
    d_id_t id;
};
} // namespace lstr
#endif // L3STER_MESH_DOMAIN_HPP
