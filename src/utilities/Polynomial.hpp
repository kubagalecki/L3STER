// Utility for handling polynomials: computing roots, etc.
/*
 * In the following, polynomials are assumed to be in the form of:
 * p(x) = sum[i = 0 : N] ( ai * x^i ), where N is the order of the polynomial
 */

#ifndef L3STER_INCGUARD_UTIL_POLYNOMIAL_HPP
#define L3STER_INCGUARD_UTIL_POLYNOMIAL_HPP

#include "Types.h"

#include <array>

namespace lstr
{
    namespace util
    {
        //////////////////////////////////////////////////////////////////////////////////////////////
        //                                      POLYNOMIAL CLASS                                    //
        //////////////////////////////////////////////////////////////////////////////////////////////
        /*
        Class containing the polynomial coefficients
        */
        template <types::poly_o_t N>
        class Polynomial final
        {
        public:
            // Ctors & Dtors
            // Forwarding constructor for the underlying array
            template <typename ... Types>
            Polynomial(const Types& ... args) : coefs(args ...) {}

            // Access
            
			// Operations
			types::val_t eval(const types::val_t&);				// evaluate polynomial
        private:
            std::array <types::val_t, N + 1> coefs;
        }

		template <types::poly_o_t N>
		types::val_t Polynomial::eval(const types::val_t& x)
		{
			// evaluate p(x)
			types::val_t ret_val = coefs.front();
			auto current_exp = x;
			for (int i = 1; i < N i++)
			{
				ret_val += current_exp * coefs[i];
				current_exp *= x;
			}
			ret_val += current_exp * coefs.back();
		}

		// Specialization for order 0 polynomial
		template<>
		class Polynomial<0>
		{
		public:
			// Ctors & Dtors
			Polynomial()							: a0(0)		{}
			Polynomial(const types::val_t& _a0)		: a0(_a0)	{}

			// Operations
			types::val_t eval(const types::val_t&)				{ return a0; }
		private:
			types::val_t a0;
		};
    }
}

#endif      //end include guard
