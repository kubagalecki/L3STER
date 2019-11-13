// Data structures representing quadratures

#ifndef L3STER_INCGUARD_MESH_QUADRATURE_HPP
#define L3STER_INCGUARD_MESH_QUADRATURE_HPP

#include "ElementTypes.h"
#include "Types.h"
#include <array>

namespace lstr
{
    namespace mesh
    {
        //////////////////////////////////////////////////////////////////////////////////////////////
        //                                    QUADRATURE BASE CLASS                                 //
        //////////////////////////////////////////////////////////////////////////////////////////////
        /*
        Base class for quadrature type
        */
        template <ElementTypes ELTYPE>
        class QuadratureBase
        {
        protected:
            static constexpr size_t getQuadratureSize(types::q_o_t);
        };

        //////////////////////////////////////////////////////////////////////////////////////////////
        //                                      QUADRATURE CLASS                                    //
        //////////////////////////////////////////////////////////////////////////////////////////////
        /*
        This class holds quadrature points and weights for a given element type and order
        */
        
        // Forward declaration of ReferenceElementBase (needed to assess the element's space dimension
        template <ElementTypes ELTYPE>
        class ReferenceElementBase;
        
        template <ElementTypes ELTYPE, types::q_o_t QORDER>
        class Quadrature final : public QuadratureBase<ELTYPE>
        {
        public:
            // Aliases
            using q_point_t = std::array <
                              std::array< types::val_t,
                              QuadratureBase<ELTYPE>::getQuadratureSize(QORDER) >,
                              ReferenceElementBase<ELTYPE>::getDim()
                              >;
            using weights_t = std::array <
                              types::val_t,
                              QuadratureBase<ELTYPE>::getQuadratureSize(QORDER)
                              >;

            // Ctors & Dtors
            Quadrature(const q_point_t&, const weights_t&);
            Quadrature();
            ~Quadrature()                                               = default;
            Quadrature(const Quadrature&)                               = default;
            Quadrature(Quadrature&&)                                    = default;
            void operator=(const Quadrature&)                           = default;
            void operator=(Quadrature&&)                                = default;

            // Access
            const q_point_t&    getQPoints()                            { return q_points; }
            const weights_t&    getWeights()                            { return weights; }
            void                setQPoints(const q_point_t& qp)         { q_points = qp; }
            void                setWeights(const weights_t& w)          { weights = w; }

        private:
            q_point_t q_points;
            weights_t weights;
        };
        
        template <ElementTypes ELTYPE, types::q_o_t QORDER>
        Quadrature<ELTYPE, QORDER>::Quadrature(const q_point_t& qpts, const weights_t& w)
        : q_points(qpts), weights(w) {}
    }
}

#endif      // end include guard
