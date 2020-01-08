// Polymorphic data structure for storing multiple elements of same type

#ifndef L3STER_INCGUARD_MESH_ELEMENTVECTOR_HPP
#define L3STER_INCGUARD_MESH_ELEMENTVECTOR_HPP

#include "mesh/Element.hpp"
#include "mesh/ElementVectorDispatcher.hpp"

#include <vector>
#include <algorithm>

namespace lstr
{
namespace mesh
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                ELEMENT VECTOR BASE CLASS                                 //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Base class for element vector.
*/
class ElementVectorBase
{
    virtual void acceptDispatcher(const ElementVectorDispatcher&) = 0;
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

    void            acceptDispatcher(const ElementVectorDispatcher& d) final { d(*this); }

private:
    // MEMBERS
    vec_t element_vector;
};
}           // namespace mesh
}           // namespace lstr

#endif      // end include guard
