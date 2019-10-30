// ReferenceElementBase class implementation

#pragma once

#include "ElementTypes.hpp"

namespace lstr
{
    namespace mesh
    {
        // Class ReferenceElementBase needs to be specialized for each element type.
        // The specializations must have the following methods:
        //
        // static constexpr size_t getNumberOfNodes(types::el_o_t)
        // static constexpr types::el_dim_t getDim()

        template <ElementTypes ELTYPE, typename CRTP_Child>
        class ReferenceElementBase;

        // QUAD
        template <typename CRTP_Child>
        class ReferenceElementBase<ElementTypes::Quad, CRTP_Child>
        {
        protected:
            static constexpr size_t getNumberOfNodes(types::el_o_t elorder)
            {
                return (elorder + 1) * (elorder + 1);
            }

            static constexpr types::el_dim_t getDim()
            {
                return 2;
            }
        };
    }
}
