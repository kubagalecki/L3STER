// Quadrature generation functionality

#ifndef L3STER_INCGUARD_QUAD_QUADRATUREGENERATOR_HPP
#define L3STER_INCGUARD_QUAD_QUADRATUREGENERATOR_HPP

#include "QuadratureTypes.h"
#include "ElementTypes.h"
#include "Types.h"
#include "Eigen/Dense"      // Eigen needed for small scale matrix computations

#include <array>

namespace lstr
{
    namespace quad
    {
        // Forward declare Quadrature class
        template <mesh::ElementTypes, QuadratureTypes, types::q_o_t>
        class Quadrature;

        template <mesh::ElementTypes ELTYPE, QuadratureTypes QTYPE, types::q_o_t QORDER>
        class QuadratureGenerator;

        template <types::q_o_t QORDER>
        class QuadratureGenerator<mesh::ElementTypes::Quad, QuadratureTypes::GLeg, QORDER>
        {
        public:
            static typename Quadrature<mesh::ElementTypes::Quad, QuadratureTypes::GLeg, QORDER>::q_points_t  getQPoints();
            static typename Quadrature<mesh::ElementTypes::Quad, QuadratureTypes::GLeg, QORDER>::weights_t   getWeights();
        };


        template <types::q_o_t QORDER>
        typename Quadrature<mesh::ElementTypes::Quad, QuadratureTypes::GLeg, QORDER>::q_points_t
        QuadratureGenerator<mesh::ElementTypes::Quad, QuadratureTypes::GLeg, QORDER>::getQPoints()
        {
            // Compute 1D Gauss-Legendre quadrature exact for order QORDER
            constexpr size_t    gl1d_size       = QORDER / 2 + 1;
            using               mat_t           = Eigen::Matrix <types::val_t, gl1d_size, gl1d_size>;

            std::array <types::val_t, gl1d_size>                gl1d_qp, gl1d_w;
            mat_t                                               gl1d_mat;

            for (auto i = 0; i < gl1d_size; ++i)
            {
                for (auto j = 0; j < gl1d_size; ++j)
                    gl1d_mat(i, j) = 0.;
            }

            for (auto i = 0; i < gl1d_size - 1; ++i)
            {
                auto I = static_cast<types::val_t>(i + 1);
                types::val_t temp = I / sqrt(4. * I * I - 1);
                gl1d_mat(i + 1, i) = temp;
                gl1d_mat(i, i + 1) = temp;
            }

            Eigen::SelfAdjointEigenSolver<mat_t> es;
            es.compute(gl1d_mat);

            typename Quadrature<mesh::ElementTypes::Quad, QuadratureTypes::GLeg, QORDER>::q_points_t ret_val;

            for (auto i = 0; i < gl1d_size; ++i)
            {
                for (auto j = 0; j < gl1d_size; ++j)
                {
                    ret_val[0][i + j * gl1d_size] = es.eigenvalues()[i];
                    ret_val[1][j + i * gl1d_size] = es.eigenvalues()[i];
                }
            }

            return ret_val;
        }

        template <types::q_o_t QORDER>
        typename Quadrature<mesh::ElementTypes::Quad, QuadratureTypes::GLeg, QORDER>::weights_t
        QuadratureGenerator<mesh::ElementTypes::Quad, QuadratureTypes::GLeg, QORDER>::getWeights()
        {
            // Compute 1D Gauss-Legendre quadrature exact for order QORDER
            constexpr size_t    gl1d_size       = QORDER / 2 + 1;
            using               mat_t           = Eigen::Matrix <types::val_t, gl1d_size, gl1d_size>;

            std::array <types::val_t, gl1d_size>                gl1d_qp, gl1d_w;
            mat_t                                               gl1d_mat;

            for (auto i = 0; i < gl1d_size; ++i)
            {
                for (auto j = 0; j < gl1d_size; ++j)
                    gl1d_mat(i, j) = 0.;
            }

            for (auto i = 0; i < gl1d_size - 1; ++i)
            {
                auto I = static_cast<types::val_t>(i + 1);
                types::val_t temp = I / sqrt(4. * I * I - 1);
                gl1d_mat(i + 1, i) = temp;
                gl1d_mat(i, i + 1) = temp;
            }

            Eigen::SelfAdjointEigenSolver<mat_t> es;
            es.compute(gl1d_mat);

            typename Quadrature<mesh::ElementTypes::Quad, QuadratureTypes::GLeg, QORDER>::weights_t ret_val;

            for (auto i = 0; i < gl1d_size; ++i)
            {
                for (auto j = 0; j < gl1d_size; ++j)
                    ret_val[i + j * gl1d_size] =
                        4 * es.eigenvectors()(0, i) * es.eigenvectors()(0, i) * es.eigenvectors()(0, j) * es.eigenvectors()(0, j);
            }

            return ret_val;
        }

    }
}

#endif      // end include guard
