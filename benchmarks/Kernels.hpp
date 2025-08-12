#include "Common.hpp"

inline constexpr auto ns3d_kernel = [](const auto& in, auto& out) {
    const auto& [vals, ders, point]             = in;
    const auto& [u, v, w, p, ox, oy, oz]        = vals;
    const auto& [x_ders, y_ders, z_ders]        = ders;
    const auto& [ux, vx, wx, px, oxx, oyx, ozx] = x_ders;
    const auto& [uy, vy, wy, py, oxy, oyy, ozy] = y_ders;
    const auto& [uz, vz, wz, pz, oxz, oyz, ozz] = z_ders;

    auto& [operators, rhs] = out;
    auto& [A0, A1, A2, A3] = operators;

    constexpr double Re_inv = 1e-3;

    A0(0, 0) = ux;
    A0(0, 1) = uy;
    A0(0, 2) = uz;
    A0(1, 0) = vx;
    A0(1, 1) = vy;
    A0(1, 2) = vz;
    A0(2, 0) = wx;
    A0(2, 1) = wy;
    A0(2, 2) = wz;
    A0(3, 4) = 1.;
    A0(4, 5) = 1.;
    A0(5, 6) = 1.;

    A1(0, 0) = u;
    A1(0, 3) = 1.;
    A1(1, 1) = u;
    A1(1, 6) = -Re_inv;
    A1(2, 2) = u;
    A1(2, 5) = Re_inv;
    A1(4, 2) = -1.;
    A1(5, 1) = 1.;
    A1(6, 0) = 1.;
    A1(7, 4) = 1.;

    A2(0, 0) = v;
    A2(0, 3) = 1.;
    A2(0, 6) = Re_inv;
    A2(1, 1) = v;
    A2(2, 2) = v;
    A2(2, 4) = -Re_inv;
    A2(3, 2) = 1.;
    A2(5, 0) = -1.;
    A2(6, 1) = 1.;
    A2(7, 5) = 1.;

    A3(0, 0) = w;
    A3(0, 3) = 1.;
    A3(0, 5) = -Re_inv;
    A3(1, 1) = w;
    A3(1, 4) = Re_inv;
    A3(2, 2) = w;
    A3(3, 1) = -1.;
    A3(4, 0) = 1.;
    A3(6, 2) = 1.;
    A3(7, 6) = 1.;

    rhs[0] = u * ux + v * uy + w * uz;
    rhs[1] = u * vx + v * vy + w * vz;
    rhs[2] = u * wx + v * wy + w * wz;
};

inline constexpr auto diff2d_kernel = [](const auto&, auto& out) noexcept {
    auto& [operators, rhs] = out;
    auto& [A0, Ax, Ay]     = operators;

    constexpr double lambda = 1.;

    A0(1, 1) = -1.;
    A0(2, 2) = -1.;

    Ax(0, 1) = -lambda;
    Ax(1, 0) = 1.;
    Ax(3, 2) = 1.;

    Ay(0, 2) = -lambda;
    Ay(2, 0) = 1.;
    Ay(3, 1) = -1.;
};

inline constexpr auto diff3d_kernel = [](const auto&, auto& out) {
    auto& [operators, rhs] = out;
    auto& [A0, Ax, Ay, Az] = operators;

    constexpr double k = 1.; // diffusivity
    constexpr double s = 1.; // source

    // -k * div q = s
    Ax(0, 1) = -k;
    Ay(0, 2) = -k;
    Az(0, 3) = -k;
    rhs[0]   = s;

    // grad T = q
    A0(1, 1) = -1.;
    Ax(1, 0) = 1.;
    A0(2, 2) = -1.;
    Ay(2, 0) = 1.;
    A0(3, 3) = -1.;
    Az(3, 0) = 1.;

    // rot q = 0
    Ay(4, 3) = 1.;
    Az(4, 2) = -1.;
    Ax(5, 3) = -1.;
    Az(5, 1) = 1.;
    Ax(6, 2) = 1.;
    Ay(6, 1) = -1.;
};
