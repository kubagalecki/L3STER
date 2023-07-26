#ifndef L3STER_COMMON_STRUCTS_HPP
#define L3STER_COMMON_STRUCTS_HPP

#include "l3ster/common/Typedefs.h"
#include "l3ster/util/EigenUtils.hpp"

namespace lstr
{
template < dim_t DIM >
    requires(DIM <= 3)
class Point
{
public:
    using coords_array_t = Eigen::Vector< val_t, DIM >;

    Point() = default;
    Point(const coords_array_t& coords_) noexcept : coords{coords_} {}
    Point(const std::array< val_t, DIM >& a) noexcept : coords{a.data()} {}
    template < std::same_as< val_t >... T >
    Point(T... coords_) : coords{std::array{coords_...}.data()}
    {}

    [[nodiscard]] val_t  operator[](ptrdiff_t i) const noexcept { return coords.coeff(i); }
    [[nodiscard]] val_t& operator[](ptrdiff_t i) noexcept { return coords.coeffRef(i); }

    [[nodiscard]] val_t x() const noexcept
        requires(DIM >= 1)
    {
        return coords.coeff(0);
    }
    [[nodiscard]] val_t y() const noexcept
        requires(DIM >= 2)
    {
        return coords.coeff(1);
    }
    [[nodiscard]] val_t z() const noexcept
        requires(DIM >= 3)
    {
        return coords.coeff(2);
    }
    [[nodiscard]] val_t& x() noexcept
        requires(DIM >= 1)
    {
        return coords.coeffRef(0);
    }
    [[nodiscard]] val_t& y() noexcept
        requires(DIM >= 2)
    {
        return coords.coeffRef(1);
    }
    [[nodiscard]] val_t& z() noexcept
        requires(DIM >= 3)
    {
        return coords.coeffRef(2);
    }

    operator coords_array_t() const noexcept { return coords; }; // NOLINT implicit conversion intended

private:
    coords_array_t coords;
};

template < size_t DIM >
Point(const std::array< val_t, DIM >&) -> Point< DIM >;

template < std::same_as< val_t >... T >
Point(T...) -> Point< sizeof...(T) >;

template < dim_t DIM >
bool operator==(const Point< DIM >& p1, const Point< DIM >& p2)
{
    return static_cast< Point< DIM >::coords_array_t >(p1) == static_cast< Point< DIM >::coords_array_t >(p2);
}

struct SpaceTimePoint
{
    Point< 3 > space;
    val_t      time;
};
} // namespace lstr
#endif // L3STER_COMMON_STRUCTS_HPP
