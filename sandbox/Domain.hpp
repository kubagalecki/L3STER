#pragma once

#include "ElementVector.hpp"
#include <map>
#include <memory>
#include <utility>

namespace lstr
{
	namespace mesh
	{
		//////////////////////////////////////////////////////////////////////////////////////////////
		//  									DOMAIN CLASS										//
		//////////////////////////////////////////////////////////////////////////////////////////////
		/*
		The domain class stores 1+ element vectors
		*/
		class Domain
		{
		public:
			using map_t = std::map< std::pair<ElementTypes, types::el_o_t>, std::unique_ptr<ElementVectorBase> >;

			types::d_id_t		getId()							{ return id; }
			void				setId(types::d_id_t _id)		{ id = _id; }

		private:
			types::d_id_t		id			= 0;
			map_t				elements;
		};
	}
}