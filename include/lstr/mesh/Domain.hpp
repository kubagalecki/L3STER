#ifndef L3STER_MESH_DOMAIN_HPP
#define L3STER_MESH_DOMAIN_HPP

#include "lstr/mesh/Aliases.hpp"
#include "lstr/mesh/Element.hpp"

#include <algorithm>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace lstr::mesh
{
class Domain
{
public:
    template < ElementTypes ELTYPE, types::el_o_t ELORDER >
    using element_vector_t = std::vector< Element< ELTYPE, ELORDER > >;
    using element_vector_variant_t =
        parametrize_type_over_element_types_and_orders_t< std::variant, element_vector_t >;
    using element_vector_variant_vector_t = std::vector< element_vector_variant_t >;

    template < ElementTypes ELTYPE, types::el_o_t ELORDER >
    void pushBack(const Element< ELTYPE, ELORDER >& element);

    template < ElementTypes ELTYPE, types::el_o_t ELORDER, typename... Args >
    void emplaceBack(Args&&... args);

    template < ElementTypes ELTYPE, types::el_o_t ELORDER >
    void reserve(size_t size);

    template < typename F >
    void visit(F& element_visitor);

    template < typename F >
    void cvisit(F& element_visitor) const;

    template < typename F >
    [[nodiscard]] std::optional< element_ref_variant_t > findElement(const F& predicate);

    template < typename F >
    [[nodiscard]] std::optional< element_cref_variant_t > findElement(const F& predicate) const;

    [[nodiscard]] types::dim_t  getDim() const { return dim; };
    [[nodiscard]] inline size_t getNElements() const;

private:
    element_vector_variant_vector_t element_vectors;
    types::dim_t                    dim = 0;

    template < typename F >
    [[nodiscard]] static auto wrapElementVisitor(F&& element_visitor);

    template < typename F >
    [[nodiscard]] static auto wrapCElementVisitor(F&& element_visitor);
};

template < ElementTypes ELTYPE, types::el_o_t ELORDER >
void Domain::pushBack(const Element< ELTYPE, ELORDER >& element)
{
    emplaceBack< ELTYPE, ELORDER >(element);
}

template < ElementTypes ELTYPE, types::el_o_t ELORDER, typename... ArgTypes >
void Domain::emplaceBack(ArgTypes&&... Args)
{
    using el_vec_t = element_vector_t< ELTYPE, ELORDER >;

    if (!element_vectors.empty())
    {
        if (ElementTraits< Element< ELTYPE, ELORDER > >::native_dim != dim)
            throw std::invalid_argument("Emplacing element in domain of different dimension");
    }
    else
        dim = ElementTraits< Element< ELTYPE, ELORDER > >::native_dim;

    const auto vector_variant_it =
        std::find_if(element_vectors.begin(), element_vectors.end(), [](const auto& v) {
            return std::holds_alternative< el_vec_t >(v);
        });
    if (vector_variant_it == element_vectors.end())
    {
        std::get< el_vec_t >(element_vectors.emplace_back(std::in_place_type< el_vec_t >))
            .emplace_back(std::forward< ArgTypes >(Args)...);
    }
    else
    {
        std::get< el_vec_t >(*vector_variant_it).emplace_back(std::forward< ArgTypes >(Args)...);
    }
}

template < ElementTypes ELTYPE, types::el_o_t ELORDER >
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
        std::find_if(element_vectors.begin(), element_vectors.end(), [](const auto& v) {
            return std::holds_alternative< el_vec_t >(v);
        });
    if (vector_variant_it == element_vectors.end())
    {
        std::get< el_vec_t >(element_vectors.emplace_back(std::in_place_type< el_vec_t >))
            .reserve(size);
    }
    else
    {
        std::get< el_vec_t >(*vector_variant_it).reserve(size);
    }
}

template < typename F >
void Domain::visit(F& element_visitor)
{
    std::for_each(element_vectors.begin(),
                  element_vectors.end(),
                  [visitor = wrapElementVisitor(element_visitor)](
                      element_vector_variant_t& el_vec) mutable { std::visit(visitor, el_vec); });
}

template < typename F >
void Domain::cvisit(F& element_visitor) const
{
    std::for_each(
        element_vectors.cbegin(),
        element_vectors.cend(),
        [visitor = wrapElementVisitor(element_visitor)](
            const element_vector_variant_t& el_vec) mutable { std::visit(visitor, el_vec); });
}

template < typename F >
std::optional< element_ref_variant_t > Domain::findElement(const F& predicate)
{
    static_assert(is_invocable_r_on_all_elements_v< F, bool >);

    std::optional< element_ref_variant_t > ret_val;

    const auto element_vector_visitor = [&ret_val, &predicate](const auto& element_vector) {
        const auto found_iter =
            std::find_if(element_vector.cbegin(), element_vector.cend(), std::cref(predicate));

        if (found_iter != element_vector.cend())
        {
            using element_t     = typename std::decay_t< decltype(element_vector) >::value_type;
            using element_ref_t = std::reference_wrapper< element_t >;

            ret_val.emplace(std::in_place_type< element_ref_t >,
                            std::ref(const_cast< element_t& >(*found_iter)));

            // Note: const_cast is used correctly, because the underlying element is not const.
            // However, we need to pass it to the predicate as a const reference, since the
            // predicate is not allowed to modify the element.
        }
    };

    for (const auto& element_vector_variant : element_vectors)
    {
        std::visit(element_vector_visitor, element_vector_variant);

        if (ret_val)
            break;
    }

    return ret_val;
}

template < typename F >
std::optional< element_cref_variant_t > Domain::findElement(const F& predicate) const
{
    // The non-const version of findElement does not modify `this`, it merely returns a non-const
    // reference to the underlying data. For this reason, we can call it as const, as long as we
    // const-qualify the returned value. Therefore, the following line of code does NOT invoke UB.
    auto nonconst_ref_opt = const_cast< Domain* >(this)->findElement(predicate);

    if (!nonconst_ref_opt)
        return std::optional< element_cref_variant_t >{};

    const auto const_qualify = [](const auto& element_ref) {
        using element_t              = typename std::decay_t< decltype(element_ref) >::type;
        constexpr auto element_type  = ElementTraits< element_t >::element_type;
        constexpr auto element_order = ElementTraits< element_t >::element_order;

        return std::optional< element_cref_variant_t >{
            std::in_place,
            std::in_place_type< element_cref_t< element_type, element_order > >,
            std::cref(element_ref)};
    };

    return std::visit(const_qualify, *nonconst_ref_opt);
}

template < typename F >
auto Domain::wrapElementVisitor(F&& element_visitor)
{
    static_assert(is_invocable_on_all_elements_v< F >);

    return [&element_visitor](auto& element_vector) {
        std::for_each(element_vector.begin(), element_vector.end(), std::ref(element_visitor));
    };
}

template < typename F >
auto Domain::wrapCElementVisitor(F&& element_visitor)
{
    static_assert(is_invocable_on_all_elements_v< F >);

    return [&element_visitor](const auto& element_vector) {
        std::for_each(element_vector.cbegin(), element_vector.cend(), std::ref(element_visitor));
    };
}

inline size_t Domain::getNElements() const
{
    return std::accumulate(
        element_vectors.cbegin(),
        element_vectors.cend(),
        0u,
        [](size_t sum, const auto& el_vec_var) {
            return sum + std::visit([](const auto& el_vec) { return el_vec.size(); }, el_vec_var);
        });
}

class DomainView
{
    using domain_ref_t = std::reference_wrapper< const Domain >;

public:
    DomainView()                  = delete;
    DomainView(const DomainView&) = default;
    DomainView(DomainView&&)      = default;
    DomainView& operator=(const DomainView&) = delete;
    DomainView& operator=(DomainView&&) = delete;
    DomainView(const Domain& domain_, types::el_id_t id_) : domain{std::cref(domain_)}, id{id_} {}

    [[nodiscard]] types::el_id_t getID() const { return id; }
    [[nodiscard]] types::dim_t   getDim() const { return domain.get().getDim(); }
    [[nodiscard]] size_t         getNElements() const { return domain.get().getNElements(); }
    [[nodiscard]] domain_ref_t   getDomainRef() const { return domain.get(); }

private:
    domain_ref_t   domain;
    types::el_id_t id;
};

} // namespace lstr::mesh

#endif // L3STER_MESH_DOMAIN_HPP
