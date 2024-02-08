#ifndef L3STER_COMMON_STRUCTS_HPP
#define L3STER_COMMON_STRUCTS_HPP

#include "l3ster/common/Typedefs.h"
#include "l3ster/util/EigenUtils.hpp"

namespace lstr
{
template < dim_t DIM >
struct Point
{
    static_assert(DIM <= 3);

    constexpr Point() = default;
    constexpr Point(const std::array< val_t, DIM >& coords_) : coords{coords_} {}
    template < std::same_as< val_t >... Coord >
    constexpr explicit Point(Coord... coords_)
        requires(sizeof...(Coord) == DIM)
        : coords{coords_...}
    {}
    constexpr operator const std::array< val_t, DIM >&() const { return coords; }
    constexpr operator std::array< val_t, DIM >() { return coords; }

    [[nodiscard]] auto asEigenVec() const -> Eigen::Map< const Eigen::Vector< val_t, static_cast< int >(DIM) > >
    {
        return Eigen::Map< const Eigen::Vector< val_t, static_cast< int >(DIM) > >{coords.data()};
    }

    constexpr val_t  operator[](ptrdiff_t i) const { return coords[i]; }
    constexpr val_t& operator[](ptrdiff_t i) { return coords[i]; }

    [[nodiscard]] constexpr val_t x() const
        requires(DIM >= 1)
    {
        return coords[0];
    }
    [[nodiscard]] constexpr val_t y() const
        requires(DIM >= 2)
    {
        return coords[1];
    }
    [[nodiscard]] constexpr val_t z() const
        requires(DIM >= 3)
    {
        return coords[2];
    }
    [[nodiscard]] constexpr val_t& x()
        requires(DIM >= 1)
    {
        return coords[0];
    }
    [[nodiscard]] constexpr val_t& y()
        requires(DIM >= 2)
    {
        return coords[1];
    }
    [[nodiscard]] constexpr val_t& z()
        requires(DIM >= 3)
    {
        return coords[2];
    }

    friend constexpr bool operator==(const Point&, const Point&)  = default;
    friend constexpr auto operator<=>(const Point&, const Point&) = default;

    std::array< val_t, DIM > coords{};
};

template < size_t DIM >
Point(const std::array< val_t, DIM >&) -> Point< DIM >;

template < std::same_as< val_t >... T >
Point(T...) -> Point< sizeof...(T) >;

struct SpaceTimePoint
{
    Point< 3 > space;
    val_t      time;
};
} // namespace lstr
#endif // L3STER_COMMON_STRUCTS_HPP
