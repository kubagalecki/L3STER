#ifndef L3STER_QUAD_QUADRATURE_HPP
#define L3STER_QUAD_QUADRATURE_HPP

#include "defs/Typedefs.h"

#include <array>

namespace lstr
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                      QUADRATURE CLASS                                    //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
This class holds quadrature points and weights for a given element type and order
*/

template < q_l_t QLENGTH, dim_t QDIM >
class Quadrature
{
public:
    using q_points_t = std::array< std::array< val_t, QDIM >, QLENGTH >;
    using weights_t  = std::array< val_t, QLENGTH >;

    static constexpr q_l_t size = QLENGTH;
    static constexpr dim_t dim  = QDIM;

    Quadrature() = default;
    Quadrature(const q_points_t&, const weights_t&);

    const q_points_t& getQPoints() const { return q_points; }
    const weights_t&  getWeights() const { return weights; }
    void              setQPoints(const q_points_t& qp) { q_points = qp; }
    void              setWeights(const weights_t& w) { weights = w; }

private:
    q_points_t q_points;
    weights_t  weights;
};

template < q_l_t QLENGTH, dim_t QDIM >
Quadrature< QLENGTH, QDIM >::Quadrature(const q_points_t& qpts, const weights_t& w) : q_points(qpts), weights(w)
{}
} // namespace lstr

#endif // L3STER_QUAD_QUADRATURE_HPP
