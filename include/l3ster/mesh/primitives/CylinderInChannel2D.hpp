#ifndef L3STER_MESH_CYLINDERINCHANNEL2D_HPP
#define L3STER_MESH_CYLINDERINCHANNEL2D_HPP

#include "l3ster/mesh/MeshUtils.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/util/Common.hpp"

namespace lstr::mesh
{
struct CylinderInChannel2DMeshIds
{
    d_id_t domain = 0, bottom = 1, top = 2, left = 3, right = 4, cylinder = 5;
};

struct CylinderInChannel2DGeometry
{
    val_t  r_inner = .5, r_outer = 2., left_offset = 10., right_offset = 16., bottom_offset = 15., top_offset = 15.;
    size_t n_circumf = 64, n_radial = 19, n_left = 8, n_right = 50, n_bottom = 15, n_top = 15;
    val_t  q_radial = 1.135, q_left = 1.3, q_right = 1.01, q_bottom = 1.2, q_top = 1.2;
    bool   quadratic = false;

    [[nodiscard]] bool correct() const
    {
        const auto geom_correct =
            r_inner < r_outer and std::min({left_offset, right_offset, bottom_offset, top_offset}) > r_outer;
        const auto discr_correct =
            n_circumf % 8 == 0 and std::min({n_circumf, n_radial, n_left, n_right, n_bottom, n_top}) > 0;
        const auto q_correct = std::min({q_radial, q_left, q_right, q_bottom, q_top}) > 0.;
        return geom_correct and discr_correct and q_correct;
    }
};

inline auto makeCylinderInChannel2DMesh(const CylinderInChannel2DGeometry& geometry = {},
                                        const CylinderInChannel2DMeshIds&  ids      = {}) -> MeshPartition< 1 >
{
    util::throwingAssert(geometry.correct());

    using std::numbers::pi;
    constexpr auto rotate = [](Point< 3 > p, val_t angle) -> Point< 3 > {
        const auto s = std::sin(angle);
        const auto c = std::cos(angle);
        return {c * p.x() - s * p.y(), s * p.x() + c * p.y(), 0.};
    };

    const auto make_cylinder = [&] {
        const auto radial_dist =
            util::geomSpaceProg(geometry.r_inner, geometry.r_outer, geometry.n_radial + 1, geometry.q_radial);
        const auto circumf_dist = util::linspace(0., 1., geometry.n_circumf / 2 + 1);
        const auto cyl_ids      = SquareMeshIds{.domain = ids.domain, .left = ids.cylinder};
        auto       top_half     = geometry.quadratic ? makeSquareMeshQuadratic(radial_dist, circumf_dist, cyl_ids)
                                                     : makeSquareMesh(radial_dist, circumf_dist, cyl_ids);
        const auto deform_half  = [&](Point< 3 > p) -> Point< 3 > {
            const auto r     = p.x();
            const auto angle = p.y() * pi;
            return {r * std::cos(angle), r * std::sin(angle), 0.};
        };
        deform(top_half, deform_half);
        auto bot_half = copy(top_half);
        deform(bot_half, std::bind_back(rotate, pi));
        return merge(top_half, bot_half);
    };
    auto cylinder = make_cylinder();

    const auto make_wake = [&](val_t L, size_t n, val_t q, d_id_t edge_id) {
        const auto x_dist   = util::geomSpaceProg(0., L, n + 1, q);
        const auto y_dist   = util::linspace(-pi / 4., pi / 4., geometry.n_circumf / 4 + 1);
        const auto wake_ids = SquareMeshIds{.domain = ids.domain, .right = edge_id};
        auto       retval   = geometry.quadratic ? makeSquareMeshQuadratic(x_dist, y_dist, wake_ids)
                                                 : makeSquareMesh(x_dist, y_dist, wake_ids);
        deform(retval, [&](Point< 3 > p) -> Point< 3 > {
            const auto x = p.x() + (1. - p.x() / L) * geometry.r_outer * std::cos(p.y());
            const auto y = geometry.r_outer * std::sin(p.y());
            return {x, y, 0.};
        });
        return retval;
    };

    const auto make_corner = [&](val_t Lx, size_t nx, val_t qx, val_t Ly, size_t ny, val_t qy, d_id_t rid, d_id_t tid) {
        const auto sqr22 = std::sqrt(2.) / 2.;
        return makeSquareMesh(util::geomSpaceProg(0., Lx, nx + 1, qx) | std::views::transform([&](val_t x) {
                                  return x + (1. - x / Lx) * geometry.r_outer * sqr22;
                              }),
                              util::geomSpaceProg(0., Ly, ny + 1, qy) | std::views::transform([&](val_t y) {
                                  return y + (1. - y / Ly) * geometry.r_outer * sqr22;
                              }),
                              {.domain = ids.domain, .top = tid, .right = rid});
    };

    auto right_wake = make_wake(geometry.right_offset, geometry.n_right, geometry.q_right, ids.right);
    auto left_wake  = make_wake(geometry.left_offset, geometry.n_left, geometry.q_left, ids.left);
    auto bot_wake   = make_wake(geometry.bottom_offset, geometry.n_bottom, geometry.q_bottom, ids.bottom);
    auto top_wake   = make_wake(geometry.top_offset, geometry.n_top, geometry.q_top, ids.top);
    deform(top_wake, std::bind_back(rotate, pi / 2.));
    deform(left_wake, std::bind_back(rotate, pi));
    deform(bot_wake, std::bind_back(rotate, 1.5 * pi));

    auto tr_corner = make_corner(geometry.right_offset,
                                 geometry.n_right,
                                 geometry.q_right,
                                 geometry.top_offset,
                                 geometry.n_top,
                                 geometry.q_top,
                                 ids.right,
                                 ids.top);
    auto tl_corner = make_corner(geometry.top_offset,
                                 geometry.n_top,
                                 geometry.q_top,
                                 geometry.left_offset,
                                 geometry.n_left,
                                 geometry.q_left,
                                 ids.top,
                                 ids.left);
    auto bl_corner = make_corner(geometry.left_offset,
                                 geometry.n_left,
                                 geometry.q_left,
                                 geometry.bottom_offset,
                                 geometry.n_bottom,
                                 geometry.q_bottom,
                                 ids.left,
                                 ids.bottom);
    auto br_corner = make_corner(geometry.bottom_offset,
                                 geometry.n_bottom,
                                 geometry.q_bottom,
                                 geometry.right_offset,
                                 geometry.n_right,
                                 geometry.q_right,
                                 ids.bottom,
                                 ids.right);
    deform(tl_corner, std::bind_back(rotate, pi / 2.));
    deform(bl_corner, std::bind_back(rotate, pi));
    deform(br_corner, std::bind_back(rotate, 1.5 * pi));

    cylinder = merge(cylinder, right_wake);
    cylinder = merge(cylinder, left_wake);
    cylinder = merge(cylinder, bot_wake);
    cylinder = merge(cylinder, top_wake);
    cylinder = merge(cylinder, tr_corner);
    cylinder = merge(cylinder, tl_corner);
    cylinder = merge(cylinder, bl_corner);
    cylinder = merge(cylinder, br_corner);

    return cylinder;
}
} // namespace lstr::mesh
#endif // L3STER_MESH_CYLINDERINCHANNEL2D_HPP
