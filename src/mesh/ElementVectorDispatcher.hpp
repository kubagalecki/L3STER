#ifndef L3STER_INCGUARD_MESH_ELEMENTVECTORDISPATCHER_HPP
#define L3STER_INCGUARD_MESH_ELEMENTVECTORDISPATCHER_HPP

#include "typedefs/Types.h"
#include "mesh/ElementTypes.h"
#include "mesh/ElementVectorOperator.hpp"

#include <memory>

namespace lstr::mesh
{
	// Forward declare the ElementVector class
	template <ElementTypes ELTYPE, types::el_o_t ELORDER>
	class ElementVector;

	//////////////////////////////////////////////////////////////////////////////////////////////
	//                              ELEMENT VECTOR DISPATCHER CLASS								//
	//////////////////////////////////////////////////////////////////////////////////////////////
	/*
	ElementVectorDispatcher is intended for dispatching operations onto element vectors of
	different types held within a larger Domain structure
	*/
	class ElementVectorDispatcher final
	{
	public:
		template <ElementTypes ELTYPE, types::el_o_t ELORDER>
		auto operator()(ElementVector<ELTYPE, ELORDER>) const;

	private:
		std::unique_ptr<ElementVectorOperatorBase> op_ptr;
	};

	template <ElementTypes ELTYPE, types::el_o_t ELORDER>
	auto ElementVectorDispatcher::operator()(ElementVector<ELTYPE, ELORDER> v) const
	{
		return static_cast< ElementVectorOperator<ELTYPE, ELORDER>* >(op_ptr.get())->operator()(v);
	}
}

#endif      // end include guard
