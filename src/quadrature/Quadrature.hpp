// Data structures representing quadratures

#ifndef L3STER_INCGUARD_QUAD_QUADRATURE_HPP
#define L3STER_INCGUARD_QUAD_QUADRATURE_HPP

#include "mesh/ElementTypes.hpp"
#include "quadrature/QuadratureTypes.h"
#include "typedefs/Types.h"

#include <array>

namespace lstr {
    namespace quad {
//////////////////////////////////////////////////////////////////////////////////////////////
//                                    QUADRATURE BASE CLASS                                 //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Empty base class for quadrature type
*/
        class QuadratureBase {
        };

//////////////////////////////////////////////////////////////////////////////////////////////
//                                      QUADRATURE CLASS                                    //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
This class holds quadrature points and weights for a given element type and order
*/

        template<types::q_l_t QLENGTH, types::dim_t QDIM>
        class Quadrature final : public QuadratureBase {
        public:
            // Aliases
            using q_points_t = std::array<std::array<types::val_t, QLENGTH>, QDIM>;
            using weights_t = std::array<types::val_t, QLENGTH>;

            // Ctors & Dtors
            Quadrature() = default;

            Quadrature(const Quadrature &) = default;

            Quadrature(Quadrature &&) = default;

            Quadrature &operator=(const Quadrature &) = default;

            Quadrature &operator=(Quadrature &&) = default;

            Quadrature(const q_points_t &, const weights_t &);

            // Access
            const q_points_t &getQPoints() { return q_points; }

            const weights_t &getWeights() { return weights; }

            void setQPoints(const q_points_t &qp) { q_points = qp; }

            void setWeights(const weights_t &w) { weights = w; }

        private:
            q_points_t q_points;
            weights_t weights;
        };

        template<types::q_l_t QLENGTH, types::dim_t QDIM>
        Quadrature<QLENGTH, QDIM>::Quadrature(const q_points_t &qpts, const weights_t &w)
                : q_points(qpts), weights(w) {
        }
    } // namespace quad
} // namespace lstr

#endif // end include guard
