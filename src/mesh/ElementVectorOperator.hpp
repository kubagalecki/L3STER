#ifndef L3STER_INCGUARD_MESH_ELEMENTVECTOROPERATOR_HPP
#define L3STER_INCGUARD_MESH_ELEMENTVECTOROPERATOR_HPP

#include "typedefs/Types.h"
#include "mesh/ElementTypes.h"

namespace lstr::mesh
{
	// Forward declare the ElementVector and base classes
	class ElementVectorBase;

	template <ElementTypes ELTYPE, types::el_o_t ELORDER>
	class ElementVector;

	//////////////////////////////////////////////////////////////////////////////////////////////
	//                            ELEMENT VECTOR OPERATOR BASE CLASS							//
	//////////////////////////////////////////////////////////////////////////////////////////////
	/*
	Empty base class for ElementVectorOperator
	*/
	class ElementVectorOperatorBase {};

	//////////////////////////////////////////////////////////////////////////////////////////////
	//                               ELEMENT VECTOR OPERATOR CLASS								//
	//////////////////////////////////////////////////////////////////////////////////////////////
	/*
	ElementVectorOperator represents operations on element vectors
	*/
	template <ElementTypes ELTYPE, types::el_o_t ELORDER>
	class ElementVectorOperator : public ElementVectorOperatorBase
	{
	public:
		virtual void operator()(ElementVector<ELTYPE, ELORDER>) const = 0;
	};
}

#endif      // end include guard
