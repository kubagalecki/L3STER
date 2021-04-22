#ifndef L3STER_MESH_DOMAIN_HPP
#define L3STER_MESH_DOMAIN_HPP

#include "mesh/Aliases.hpp"
#include "mesh/Element.hpp"

#include <algorithm>
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

    template < ElementTypes ELTYPE, el_o_t ELORDER >
    void push(const Element< ELTYPE, ELORDER >& element);
    template < ElementTypes ELTYPE, el_o_t ELORDER, typename... Args >
    void emplaceBack(Args&&... args);
    template < ElementTypes ELTYPE, el_o_t ELORDER >
    auto getBackInserter();
    template < ElementTypes ELTYPE, el_o_t ELORDER >
    void reserve(size_t size);

    template < invocable_on_elements F >
    void visit(F&& element_visitor);
    template < invocable_on_const_elements F >
    void cvisit(F&& element_visitor) const;

    template < invocable_on_const_elements_r< bool > F >
    [[nodiscard]] std::optional< element_ptr_variant_t > find(F&& predicate);
    template < invocable_on_const_elements_r< bool > F >
    [[nodiscard]] std::optional< element_cptr_variant_t > find(F&& predicate) const;

    [[nodiscard]] dim_t         getDim() const { return dim; };
    [[nodiscard]] inline size_t getNElements() const;

private:
    template < ElementTypes ELTYPE, el_o_t ELORDER >
    std::vector< Element< ELTYPE, ELORDER > >& getElementVector();

    template < typename F >
    [[nodiscard]] static auto wrapElementVisitor(F& element_visitor);

    template < typename F >
    [[nodiscard]] static auto wrapCElementVisitor(F& element_visitor);

    element_vector_variant_vector_t element_vectors;
    dim_t                           dim = 0;
};

template < ElementTypes ELTYPE, el_o_t ELORDER >
std::vector< Element< ELTYPE, ELORDER > >& Domain::getElementVector()
{
    using el_vec_t = element_vector_t< ELTYPE, ELORDER >;
    if (!element_vectors.empty())
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

template < ElementTypes ELTYPE, el_o_t ELORDER, typename... ArgTypes >
void Domain::emplaceBack(ArgTypes&&... Args)
{
    getElementVector< ELTYPE, ELORDER >().emplace_back(std::forward< ArgTypes >(Args)...);
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

    if (!element_vectors.empty())
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

template < invocable_on_elements F >
void Domain::visit(F&& element_visitor)
{
    auto visitor = wrapElementVisitor(element_visitor);
    std::ranges::for_each(element_vectors, [&](element_vector_variant_t& el_vec) { std::visit(visitor, el_vec); });
}

template < invocable_on_const_elements F >
void Domain::cvisit(F&& element_visitor) const
{
    auto visitor = wrapCElementVisitor(element_visitor);
    std::ranges::for_each(element_vectors,
                          [&](const element_vector_variant_t& el_vec) { std::visit(visitor, el_vec); });
}

template < invocable_on_const_elements_r< bool > F >
std::optional< element_ptr_variant_t > Domain::find(F&& predicate)
{
    std::optional< element_ptr_variant_t > opt_el_ptr_variant;
    const auto vector_visitor = [&]< ElementTypes T, el_o_t O >(const element_vector_t< T, O >& el_vec) {
        const auto el_it = std::ranges::find_if(el_vec, std::ref(predicate));
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

template < invocable_on_const_elements_r< bool > F >
std::optional< element_cptr_variant_t > Domain::find(F&& predicate) const
{
    const auto nonconst_ptr_opt = const_cast< Domain* >(this)->find(predicate);
    if (!nonconst_ptr_opt)
        return {};
    return constifyVariant(*nonconst_ptr_opt);
}

template < typename F >
auto Domain::wrapElementVisitor(F& element_visitor)
{
    return [&element_visitor](auto& element_vector) {
        std::ranges::for_each(element_vector, std::ref(element_visitor));
    };
}

template < typename F >
auto Domain::wrapCElementVisitor(F& element_visitor)
{
    return [&](const auto& element_vector) {
        std::ranges::for_each(element_vector, std::ref(element_visitor));
    };
}

size_t Domain::getNElements() const
{
    return std::accumulate(
        element_vectors.cbegin(), element_vectors.cend(), 0u, [](size_t sum, const auto& el_vec_var) {
            return sum + std::visit([](const auto& el_vec) { return el_vec.size(); }, el_vec_var);
        });
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
