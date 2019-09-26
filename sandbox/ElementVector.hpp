#pragma once

#include "Element.hpp"

//#include <functional>
#include <vector>
#include <algorithm>

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
			using el_t				= Element<ELTYPE, ELORDER>;
			using vec_t				= std::vector<el_t>;
			using vec_iter_t		= vec_t::iterator;
			using vec_citer_t		= vec_t::const_iterator;

			// METHODS
			const vec_t&	getConstRef()	const					{ return vec_t; }
			vec_t&			getRef()								{ return vec_t; }

			vec_citer_t		cbegin()		const					{ return element_vector.cbegin(); }
			vec_citer_t		cend()			const					{ return element_vector.cend(); }

			vec_iter_t		begin()									{ return element_vector.begin(); }
			vec_iter_t		end()									{ return element_vector.end(); }

			//void			for_each(std::function<el_t> f)			{ for_each(this->begin(), this->end(), f); }
			//void			const_for_each(std::function<el_t> f)	{ for_each(this->cbegin(), this->cend(), f); }

		private:
			// MEMBERS
			vec_t element_vector;
		};
	}
}