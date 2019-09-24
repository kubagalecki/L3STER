#pragma once

#include "Element.hpp"

#include <vector>

namespace lstr
{
	namespace mesh
	{
		class ElementVectorBase 
		{

		};

		template <ElementTypes ELTYPE, types::el_o_t ELORDER>
		class ElementVector final :public ElementVectorBase
		{
		public:
			// ALIASES
			using vec_t = std::vector<Element< ELTYPE, ELORDER >>;

			// GETTERS
			const vec_t& getConstRef() const	{ return vec_t; }
			vec_t& getRef()						{ return vec_t; }
		private:
			// MEMBERS
			vec_t element_vector;
		};
	}
}