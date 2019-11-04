// ReferenceElement class implementation

#pragma once

#include "ElementTypes.hpp"
#include "Types.h"

namespace lstr
{
    namespace mesh
    {
		//////////////////////////////////////////////////////////////////////////////////////////////
		//  						    REFERENCE ELEMENT BASE CLASS								//
		//////////////////////////////////////////////////////////////////////////////////////////////
		/*
		 Class ReferenceElementBase needs to be specialized for each element type.
		 The specializations must have the following methods:		
		 -static constexpr size_t getNumberOfNodes(types::el_o_t)
		 -static constexpr types::el_dim_t getDim()
		 */
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

		//////////////////////////////////////////////////////////////////////////////////////////////
		//  						      REFERENCE ELEMENT CLASS									//
		//////////////////////////////////////////////////////////////////////////////////////////////
		/*
		This static class constitutes the interface through which general information about elements of
		a given type and order is available. The interface is available within the constexpr context.
		*/
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
