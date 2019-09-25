#pragma once

#include "Element.hpp"

#include <vector>

namespace lstr
{
	namespace mesh
	{
		class ElementVectorBase
		{
		protected:
			// Cannot directly instantiate ElementVectorBase, only derived classes
			ElementVectorBase()											= default;
			ElementVectorBase(const ElementVectorBase&)					= default;
			ElementVectorBase& operator=(const ElementVectorBase&)		= default;
			virtual ~ElementVectorBase()								= default;
			ElementVectorBase(const ElementVectorBase&&)				= delete;
			ElementVectorBase& operator=(const ElementVectorBase&&)		= delete;
		};

		template <ElementTypes ELTYPE, types::el_o_t ELORDER>
		class ElementVector final :public ElementVectorBase
		{
		public:
			// ALIASES
			using vec_t = std::vector<Element< ELTYPE, ELORDER >>;

			// GETTERS
			const vec_t&		getConstRef() const			{ return vec_t; }
			vec_t&				getRef()					{ return vec_t; }

		private:
			// MEMBERS
			vec_t element_vector;
		};
	}
}