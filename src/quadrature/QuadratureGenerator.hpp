// Quadrature generation functionality

#ifndef L3STER_INCGUARD_QUAD_QUADRATUREGENERATOR_HPP
#define L3STER_INCGUARD_QUAD_QUADRATUREGENERATOR_HPP

#include "mesh/ElementTypes.h"
#include "typedefs/Types.h"
#include "quadrature/QuadratureTypes.h"
#include "quadrature/Quadrature.hpp"
#include "Eigen/Dense"      // Eigen needed for small scale matrix computations

#include <array>

namespace lstr
{
namespace quad
{
// Forward declare eferenceElementTraits
template <mesh::ElementTypes ELTYPE>
struct ReferenceElementTraits;

//////////////////////////////////////////////////////////////////////////////////////////////
//                                 QUADRATURE GENERATOR CLASS                               //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Helper class for generating concrete quadrature points and weights for specific quad. order,
element type, and quadrature type. To be specialized below. The specializations must include:

static typename Quadrature<ELTYPE, QTYPE, QORDER>::q_points_t  getQPoints();

static typename Quadrature<ELTYPE, QTYPE, QORDER>::weights_t   getWeights();
*/
template <mesh::ElementTypes ELTYPE>
class QuadratureGenerator;

template <>
class QuadratureGenerator<mesh::ElementTypes::Quad>
{
public:
    static constexpr types::q_l_t   getQLength(QuadratureTypes, types::q_o_t);
    static constexpr types::dim_t   getQDim(QuadratureTypes, types::q_o_t);

    template <QuadratureTypes QTYPE, types::q_o_t QORDER>
    static auto getQuadrature();
};

constexpr types::q_l_t QuadratureGenerator<mesh::ElementTypes::Quad>::
getQLength(QuadratureTypes q_type, types::q_o_t q_order)
{
    switch (q_type)
    {
    case QuadratureTypes::GLeg:
        return (q_order / 2 + 1) * (q_order / 2 + 1);
        break;

    case QuadratureTypes::GLob:
        return (q_order / 2 + 3) * (q_order / 2 + 3);
        break;
    }
}

constexpr types::q_l_t QuadratureGenerator<mesh::ElementTypes::Quad>::
getQDim(QuadratureTypes, types::q_o_t)
{
    return 2;
}

template <QuadratureTypes QTYPE, types::q_o_t QORDER>
auto QuadratureGenerator<mesh::ElementTypes::Quad>::getQuadrature()
{
    // Assert the quad. type has an implementatio.
    // Add || QTYPE == ... for future quad. types
    static_assert(QTYPE == QuadratureTypes::GLeg, "The requested quadrature type is not implemented");

    if constexpr(QTYPE == QuadratureTypes::GLeg)
    {
        constexpr size_t    gl1d_size       = QORDER / 2 + 1;
        using               mat_t           = Eigen::Matrix <types::val_t, gl1d_size, gl1d_size>;

        mat_t                                               gl1d_mat;

        for (size_t i = 0; i < gl1d_size; ++i)
        {
            for (size_t j = 0; j < gl1d_size; ++j)
                gl1d_mat(i, j) = 0.;
        }

        for (size_t i = 0; i < gl1d_size - 1; ++i)
        {
            auto I = static_cast<types::val_t>(i + 1);
            types::val_t temp = I / sqrt(4. * I * I - 1);
            gl1d_mat(i + 1, i) = temp;
            gl1d_mat(i, i + 1) = temp;
        }

        Eigen::SelfAdjointEigenSolver<mat_t> es;
        es.compute(gl1d_mat);

        typename Quadrature<QuadratureGenerator<mesh::ElementTypes::Quad>::getQLength(QTYPE, QORDER),
                 QuadratureGenerator<mesh::ElementTypes::Quad>::getQDim(QTYPE, QORDER)>::q_points_t ret_arr1;
        typename Quadrature<QuadratureGenerator<mesh::ElementTypes::Quad>::getQLength(QTYPE, QORDER),
                 QuadratureGenerator<mesh::ElementTypes::Quad>::getQDim(QTYPE, QORDER)>::weights_t ret_arr2;

        for (size_t i = 0; i < gl1d_size; ++i)
        {
            for (size_t j = 0; j < gl1d_size; ++j)
            {
                ret_arr1[0][i + j * gl1d_size] = es.eigenvalues()[i];
                ret_arr1[1][j + i * gl1d_size] = es.eigenvalues()[i];
                ret_arr2[i + j * gl1d_size] = 4 * es.eigenvectors()(0, i) * es.eigenvectors()(0, i) *
                                              es.eigenvectors()(0, j) * es.eigenvectors()(0, j);
            }
        }

        return Quadrature<QuadratureGenerator<mesh::ElementTypes::Quad>::getQLength(QTYPE, QORDER),
               QuadratureGenerator<mesh::ElementTypes::Quad>::getQDim(QTYPE, QORDER)> {ret_arr1, ret_arr2};
    }
}
}           // namespace quad
}           // namespace lstr

#endif      // end include guard
