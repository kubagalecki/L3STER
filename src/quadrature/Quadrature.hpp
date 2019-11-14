// Data structures representing quadratures

#ifndef L3STER_INCGUARD_QUAD_QUADRATURE_HPP
#define L3STER_INCGUARD_QUAD_QUADRATURE_HPP

#include "ElementTypes.h"
#include "Types.h"
#include "QuadratureGenerator.hpp"
#include "QuadratureTypes.h"

#include <array>

namespace lstr
{
    // Forward declaration of ReferenceElementBase (needed to assess the element's space dimension
    namespace mesh
    {
        template <mesh::ElementTypes ELTYPE>
        class ReferenceElementBase;
    }

    namespace quad
    {
        //////////////////////////////////////////////////////////////////////////////////////////////
        //                                    QUADRATURE BASE CLASS                                 //
        //////////////////////////////////////////////////////////////////////////////////////////////
        /*
        Base class for quadrature type
        */
        template <mesh::ElementTypes ELTYPE>
        class QuadratureBase {};

        //////////////////////////////////////////////////////////////////////////////////////////////
        //                                      QUADRATURE CLASS                                    //
        //////////////////////////////////////////////////////////////////////////////////////////////
        /*
        This class holds quadrature points and weights for a given element type and order
        */

        template <mesh::ElementTypes ELTYPE, QuadratureTypes QTYPE, types::q_o_t QORDER>
        class Quadrature final : public QuadratureBase<ELTYPE>
        {
        public:
            // Aliases
            using parent_t  = QuadratureBase<ELTYPE>;
            using q_points_t = std::array < std::array< types::val_t,
                              mesh::ReferenceElementBase<ELTYPE>::getQuadratureSize(QTYPE, QORDER) >,
                              mesh::ReferenceElementBase<ELTYPE>::getDim() >;
            using weights_t = std::array < types::val_t,
                              mesh::ReferenceElementBase<ELTYPE>::getQuadratureSize(QTYPE, QORDER) >;

            // Ctors & Dtors
            Quadrature();
            Quadrature(const q_points_t&, const weights_t&);

            // Access
            const q_points_t&    getQPoints()                           { return q_points; }
            const weights_t&    getWeights()                            { return weights; }
            void                setQPoints(const q_points_t& qp)        { q_points = qp; }
            void                setWeights(const weights_t& w)          { weights = w; }

        private:
            q_points_t q_points;
            weights_t weights;
        };

        template <mesh::ElementTypes ELTYPE, QuadratureTypes QTYPE, types::q_o_t QORDER>
        Quadrature<ELTYPE, QTYPE, QORDER>::Quadrature(const q_points_t& qpts, const weights_t& w)
            : q_points(qpts), weights(w) {}

        template <mesh::ElementTypes ELTYPE, QuadratureTypes QTYPE, types::q_o_t QORDER>
        Quadrature<ELTYPE, QTYPE, QORDER>::Quadrature() :
            q_points(QuadratureGenerator<ELTYPE, QTYPE, QORDER>::getQPoints()),
            weights(QuadratureGenerator<ELTYPE, QTYPE, QORDER>::getWeights()) {}
    }
}

#endif      // end include guard
