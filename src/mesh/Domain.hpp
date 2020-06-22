#ifndef L3STER_INCGUARD_MESH_DOMAIN_HPP
#define L3STER_INCGUARD_MESH_DOMAIN_HPP

#include "definitions/Aliases.hpp"
#include "mesh/Element.hpp"

#include <algorithm>
#include <utility>
#include <variant>
#include <vector>

namespace lstr::mesh
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                      DOMAIN CLASS                                        //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
The domain class stores element vectors
*/
class Domain
{
public:
    template < ElementTypes ELTYPE, types::el_o_t ELORDER >
    using element_vector_t = std::vector< Element< ELTYPE, ELORDER > >;
    using variant_t = parametrize_over_element_types_and_orders_t< std::variant, element_vector_t >;
    using element_vector_vector_t = std::vector< variant_t >;

    template < ElementTypes ELTYPE, types::el_o_t ELORDER >
    void pushBack(const Element< ELTYPE, ELORDER >&);

    template < ElementTypes ELTYPE, types::el_o_t ELORDER, typename... Args >
    void emplaceBack(Args&&... args);

    template < ElementTypes ELTYPE, types::el_o_t ELORDER >
    void reserve(size_t);

    template < typename F >
    void visit(const F&);

    template < typename F >
    void cvisit(const F&) const;

private:
    element_vector_vector_t element_vectors;
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
void Domain::visit(const F& fun)
{
    std::for_each(element_vectors.begin(), element_vectors.end(), [&fun](variant_t& v) {
        std::visit(fun, v);
    });
}

template < typename F >
void Domain::cvisit(const F& fun) const
{
    std::for_each(element_vectors.cbegin(), element_vectors.cend(), [&fun](const variant_t& v) {
        std::visit(fun, v);
    });
}
} // namespace lstr::mesh

#endif // L3STER_INCGUARD_MESH_DOMAIN_HPP
