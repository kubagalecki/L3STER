#pragma once

#include "Element.hpp"

//#include <functional>
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
        Uninstantiable base class for element vector.
        */
        class ElementVectorBase
        {
//         protected:
//             Cannot directly instantiate ElementVectorBase, only derived classes
//             ElementVectorBase()                                         = default;
//             ElementVectorBase(const ElementVectorBase&)                 = delete;
//             ElementVectorBase& operator=(const ElementVectorBase&)      = delete;
//             virtual ~ElementVectorBase()                                = default;
//             ElementVectorBase(const ElementVectorBase&&)                = delete;
//             ElementVectorBase& operator=(const ElementVectorBase&&)     = delete;
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
            const vec_t&    getConstRef()   const                   { return element_vector; }
            vec_t&          getRef()                                { return element_vector; }

            vec_citer_t     cbegin()        const                   { return element_vector.cbegin(); }
            vec_citer_t     cend()          const                   { return element_vector.cend(); }

            vec_iter_t      begin()                                 { return element_vector.begin(); }
            vec_iter_t      end()                                   { return element_vector.end(); }

            //void          for_each(std::function<el_t> f)         { for_each(this->begin(), this->end(), f); }
            //void          const_for_each(std::function<el_t> f)   { for_each(this->cbegin(), this->cend(), f); }

        private:
            // MEMBERS
            vec_t element_vector;
        };
    }
}
