// Quadrature generation functionality

#ifndef L3STER_INCGUARD_QUAD_QUADRATUREGENERATOR_HPP
#define L3STER_INCGUARD_QUAD_QUADRATUREGENERATOR_HPP

#include "ElementTypes.h"
#include "Types.h"

#include <array>

namespace lstr
{
    namespace quad
    {
        template <mesh::ElementTypes ELTYPE, types::q_o_t QORDER>
        class QuadratureGenerator;
        
        template <types::q_o_t QORDER>
        class QuadratureGenerator<mesh::ElementTypes::Quad, QORDER>
        {
        private:
            //std::array<double
        };
    }
}

#endif      // end include guard
