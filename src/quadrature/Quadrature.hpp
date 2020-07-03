// Data structures representing quadratures

#ifndef L3STER_INCGUARD_QUAD_QUADRATURE_HPP
#define L3STER_INCGUARD_QUAD_QUADRATURE_HPP

#include <array>

namespace lstr::quad
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                      QUADRATURE CLASS                                    //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
This class holds quadrature points and weights for a given element type and order
*/

template < types::q_l_t QLENGTH, types::dim_t QDIM >
class Quadrature
{
public:
    // Aliases
    using q_points_t = std::array< std::array< types::val_t, QLENGTH >, QDIM >;
    using weights_t  = std::array< types::val_t, QLENGTH >;

    // Ctors & Dtors
    Quadrature()                      = default;
    Quadrature(const Quadrature&)     = delete;
    Quadrature(Quadrature&&) noexcept = default;
    Quadrature& operator=(const Quadrature&) = delete;
    Quadrature& operator=(Quadrature&&) noexcept = default;
    Quadrature(const q_points_t&, const weights_t&);

    // Access
    const q_points_t& getQPoints() const { return q_points; }
    const weights_t&  getWeights() const { return weights; }
    void              setQPoints(const q_points_t& qp) { q_points = qp; }
    void              setWeights(const weights_t& w) { weights = w; }

private:
    q_points_t q_points;
    weights_t  weights;
};

template < types::q_l_t QLENGTH, types::dim_t QDIM >
Quadrature< QLENGTH, QDIM >::Quadrature(const q_points_t& qpts, const weights_t& w)
    : q_points(qpts), weights(w)
{}
} // namespace lstr::quad

#endif // end include guard
