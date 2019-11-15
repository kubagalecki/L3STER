// Utility for handling polynomials: computing roots, etc.
/*
 * In the following, polynomials are assumed to be in the form of:
 * p(x) = sum[0 - N] ( ai * x^i ), aN == 1, where N is the order of the polynomial
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
            
        private:
            std::array < types::val_t, N + 1 > coefs;
        }
    }
}

#endif      //end include guard
