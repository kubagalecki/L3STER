// Polymorphic data structure for storing multiple elements of same type

#ifndef L3STER_INCGUARD_MESH_ELEMENTVECTOR_HPP
#define L3STER_INCGUARD_MESH_ELEMENTVECTOR_HPP

#include "mesh/Element.hpp"
#include "utility/Visitor.hpp"
#include "utility/Meta.hpp"

#include <vector>
#include <algorithm>

namespace lstr::mesh
{

// Forward declare element vector class
template <ElementTypes, types::el_o_t>
class ElementVector;

// Define alias for easy templating over vectors of elements of all type/order combinations
template <template <typename ...> typename T>
using TemplateOverAllElementVectors = typename util::meta::cartesian_product_t<T,
      ElementVector, ElementTypesArray, ElementOrdersArray>::type;

//////////////////////////////////////////////////////////////////////////////////////////////
//                                ELEMENT VECTOR BASE CLASS                                 //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Base class for element vector.
*/
class ElementVectorBase
{
protected:
    using visitor_t = TemplateOverAllElementVectors<util::VisitorBase>;
    using v_ptr_t   = std::shared_ptr<visitor_t>;

private:
    // Accept visitor (C suffix implies the const variant
    // - the visitor does not alter the underlying elements)
    virtual void acceptVisitor(const v_ptr_t&)          = 0;
    virtual void acceptVisitorC(const v_ptr_t&) const   = 0;
};

//////////////////////////////////////////////////////////////////////////////////////////////
//                                  ELEMENT VECTOR CLASS                                    //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Wrapper for a std::vector of elements of a given type.
*/
template <ElementTypes ELTYPE, types::el_o_t ELORDER>
class ElementVector final : public ElementVectorBase
{
public:
    // ALIASES
    using el_t              = Element<ELTYPE, ELORDER>;
    using vec_t             = std::vector<el_t>;
    using vec_iter_t        = typename vec_t::iterator;
    using vec_citer_t       = typename vec_t::const_iterator;

    // METHODS
    vec_t&          getRef()
    {
        return element_vector;
    }

    const vec_t&    getConstRef() const
    {
        return element_vector;
    }

    void acceptVisitor(const v_ptr_t&) final;
    void acceptVisitorC(const v_ptr_t&) const final;

private:
    // MEMBERS
    vec_t element_vector;
};

template <ElementTypes ELTYPE, types::el_o_t ELORDER>
void ElementVector<ELTYPE, ELORDER>::acceptVisitor(const v_ptr_t& visitor)
{
    visitor->visit(*this);
}

template <ElementTypes ELTYPE, types::el_o_t ELORDER>
void ElementVector<ELTYPE, ELORDER>::acceptVisitorC(const v_ptr_t& visitor) const
{
    visitor->cvisit(*this);
}

}           // namespace lstr::mesh

#endif      // end include guard
