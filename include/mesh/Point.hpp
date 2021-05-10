#ifndef L3STER_MESH_POINT_HPP
#define L3STER_MESH_POINT_HPP

#include "defs/Typedefs.h"

#include "Eigen/Core"

#include <concepts>

namespace lstr
{
template < dim_t DIM >
requires(DIM <= 3) class Point
{
public:
    using vector_t = Eigen::Matrix< val_t, DIM, 1u >;

    Point() = default;
    explicit Point(const vector_t& coords_) : coords{coords_} {}            // NOLINT implicit conversion intended
    explicit Point(const std::array< val_t, DIM >& a) : coords{a.data()} {} // NOLINT implicit conversion intended

    [[nodiscard]] val_t  x() const requires(DIM >= 1) { return coords.coeff(0); }
    [[nodiscard]] val_t  y() const requires(DIM >= 2) { return coords.coeff(1); }
    [[nodiscard]] val_t  z() const requires(DIM >= 3) { return coords.coeff(2); }
    [[nodiscard]] val_t& x() requires(DIM >= 1) { return coords.coeffRef(0); }
    [[nodiscard]] val_t& y() requires(DIM >= 2) { return coords.coeffRef(1); }
    [[nodiscard]] val_t& z() requires(DIM >= 3) { return coords.coeffRef(2); }

    operator vector_t() const { return coords; }; // NOLINT implicit conversion intended

private:
    vector_t coords;
};

template < size_t DIM >
Point(const std::array< val_t, DIM >&) -> Point< DIM >;
} // namespace lstr
#endif // L3STER_MESH_POINT_HPP
