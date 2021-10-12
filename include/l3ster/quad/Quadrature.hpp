#ifndef L3STER_QUAD_QUADRATURE_HPP
#define L3STER_QUAD_QUADRATURE_HPP

#include "l3ster/defs/Typedefs.h"

#include <array>

namespace lstr
{
template < q_l_t QLENGTH, dim_t QDIM >
class Quadrature
{
public:
    using q_points_t = std::array< std::array< val_t, QDIM >, QLENGTH >;
    using weights_t  = std::array< val_t, QLENGTH >;

    static constexpr q_l_t size = QLENGTH;
    static constexpr dim_t dim  = QDIM;

    Quadrature() = default;
    Quadrature(const q_points_t& qpts, const weights_t& w) : points(qpts), weights(w) {}

    const auto& getPoints() const { return points; }
    const auto& getWeights() const { return weights; }
    void        setQPoints(const q_points_t& qp) { points = qp; }
    void        setWeights(const weights_t& w) { weights = w; }

private:
    q_points_t points;
    weights_t  weights;
};
} // namespace lstr
#endif // L3STER_QUAD_QUADRATURE_HPP
