// Polymorphic data structure for storing multiple elements of same type

#ifndef L3STER_INCGUARD_MESH_ELEMENTVECTOR_HPP
#define L3STER_INCGUARD_MESH_ELEMENTVECTOR_HPP

#include "mesh/Element.hpp"

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
        Empty base class for element vector.
        */
        class ElementVectorBase {};

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
            const vec_t&    getConstRef()   const                   { return element_vector; }
            vec_t&          getRef()                                { return element_vector; }

        private:
            // MEMBERS
            vec_t element_vector;
        };
    }
}

#endif      // end include guard
