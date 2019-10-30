// ReferenceElement class implementation

#pragma once

#include "ElementTypes.hpp"
#include "Types.h"
#include "ReferenceElementBase.hpp"

namespace lstr
{
    namespace mesh
    {
        template <ElementTypes ELTYPE, types::el_o_t ELORDER>
        class ReferenceElement final : public ReferenceElementBase<ELTYPE, ReferenceElement< ELTYPE, ELORDER >>
        {
            // ALIASES
            using parent_t = ReferenceElementBase<ELTYPE, ReferenceElement>;

            // ReferenceElement is a static class
            ReferenceElement()										= delete;
            ReferenceElement(const ReferenceElement&)				= delete;
            ReferenceElement& operator=(const ReferenceElement&)	= delete;
            virtual ~ReferenceElement()								= delete;
            ReferenceElement(const ReferenceElement&&)				= delete;
            ReferenceElement& operator=(const ReferenceElement&&)	= delete;

        public:
            // METHODS
            static constexpr size_t				getNumberOfNodes()
            {
                return parent_t::getNumberOfNodes(ELORDER);
            }

            static constexpr types::el_dim_t	getDim()
            {
                return parent_t::getDim(ELORDER);
            }
        };
    }
}
