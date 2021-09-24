#ifndef L3STER_MESH_POINT_HPP
#define L3STER_MESH_POINT_HPP

#include "l3ster/defs/Typedefs.h"

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
    explicit Point(const vector_t& coords_) noexcept : coords{coords_} {}
    explicit Point(const std::array< val_t, DIM >& a) noexcept : coords{a.data()} {}
    template < std::same_as< val_t >... T >
    explicit Point(T... coords_) : coords{std::array{coords_...}.data()}
    {}

    [[nodiscard]] val_t  operator[](ptrdiff_t i) const noexcept { return coords.coeff(i); }
    [[nodiscard]] val_t& operator[](ptrdiff_t i) noexcept { return coords.coeffRef(i); }

    [[nodiscard]] val_t  x() const noexcept requires(DIM >= 1) { return coords.coeff(0); }
    [[nodiscard]] val_t  y() const noexcept requires(DIM >= 2) { return coords.coeff(1); }
    [[nodiscard]] val_t  z() const noexcept requires(DIM >= 3) { return coords.coeff(2); }
    [[nodiscard]] val_t& x() noexcept requires(DIM >= 1) { return coords.coeffRef(0); }
    [[nodiscard]] val_t& y() noexcept requires(DIM >= 2) { return coords.coeffRef(1); }
    [[nodiscard]] val_t& z() noexcept requires(DIM >= 3) { return coords.coeffRef(2); }

    operator vector_t() const noexcept { return coords; }; // NOLINT implicit conversion intended

private:
    vector_t coords;
};

template < size_t DIM >
Point(const std::array< val_t, DIM >&) -> Point< DIM >;

template < std::same_as< val_t >... T >
Point(T...) -> Point< sizeof...(T) >;

template < dim_t DIM >
bool operator==(const Point< DIM >& p1, const Point< DIM >& p2)
{
    return static_cast< Point< DIM >::vector_t >(p1) == static_cast< Point< DIM >::vector_t >(p2);
}
} // namespace lstr
#endif // L3STER_MESH_POINT_HPP
