#ifndef L3STER_TESTS_KERNELS_HPP
#define L3STER_TESTS_KERNELS_HPP

// Lambda = 1
inline constexpr auto diffusion_kernel_2D = [](const auto&, auto& out) noexcept {
    auto& [operators, rhs] = out;
    auto& [A0, Ax, Ay]     = operators;

    constexpr double lambda = 1.;

    // -k * div q = s
    Ax(0, 1) = -lambda;
    Ay(0, 2) = -lambda;

    // grad T = q
    A0(1, 1) = -1.;
    Ax(1, 0) = 1.;
    A0(2, 2) = -1.;
    Ay(2, 0) = 1.;

    // curl q = 0
    Ax(3, 2) = 1.;
    Ay(3, 1) = -1.;
};

// Variable lambda passed as external field
inline constexpr auto diffusion_kernel_2D_var = [](const auto& in, auto& out) noexcept {
    const auto& [field_vals, field_ders, _] = in;
    const auto lambda                       = field_vals[0];
    const auto& [dx, dy]                    = field_ders;
    const auto dl_dx                        = dx[0];
    const auto dl_dy                        = dy[0];

    auto& [operators, rhs] = out;
    auto& [A0, Ax, Ay]     = operators;

    // -grad k * q - k * div q = s
    A0(0, 1) = -dl_dx;
    A0(0, 2) = -dl_dy;
    Ax(0, 1) = -lambda;
    Ay(0, 2) = -lambda;

    // grad T = q
    A0(1, 1) = -1.;
    Ax(1, 0) = 1.;
    A0(2, 2) = -1.;
    Ay(2, 0) = 1.;

    // curl q = 0
    Ax(3, 2) = 1.;
    Ay(3, 1) = -1.;
};

// Lambda = 1
inline constexpr auto diffusion_kernel_3D = [](const auto&, auto& out) noexcept {
    auto& [operators, rhs] = out;
    auto& [A0, Ax, Ay, Az] = operators;

    constexpr double lambda = 1.;

    // -k * div q = s
    Ax(0, 1) = -lambda;
    Ay(0, 2) = -lambda;
    Az(0, 3) = -lambda;

    // grad T = q
    A0(1, 1) = -1.;
    Ax(1, 0) = 1.;
    A0(2, 2) = -1.;
    Ay(2, 0) = 1.;
    A0(3, 3) = -1.;
    Az(3, 0) = 1.;

    // curl q = 0
    Ay(4, 3) = 1.;
    Az(4, 2) = -1.;
    Ax(5, 3) = -1.;
    Az(5, 1) = 1.;
    Ax(6, 2) = 1.;
    Ay(6, 1) = -1.;
};

// Variable lambda passed as external field
inline constexpr auto diffusion_kernel_3D_var = [](const auto& in, auto& out) noexcept {
    const auto& [field_vals, field_ders, _] = in;
    const auto lambda                       = field_vals[0];
    const auto& [dx, dy, dz]                = field_ders;
    const auto dl_dx                        = dx[0];
    const auto dl_dy                        = dy[0];
    const auto dl_dz                        = dz[0];

    auto& [operators, rhs] = out;
    auto& [A0, Ax, Ay, Az] = operators;

    // -grad k * q - k * div q = s
    A0(0, 1) = -dl_dx;
    A0(0, 2) = -dl_dy;
    A0(0, 3) = -dl_dz;
    Ax(0, 1) = -lambda;
    Ay(0, 2) = -lambda;
    Az(0, 3) = -lambda;

    // grad T = q
    A0(1, 1) = -1.;
    Ax(1, 0) = 1.;
    A0(2, 2) = -1.;
    Ay(2, 0) = 1.;
    A0(3, 3) = -1.;
    Az(3, 0) = 1.;

    // curl q = 0
    Ay(4, 3) = 1.;
    Az(4, 2) = -1.;
    Ax(5, 3) = -1.;
    Az(5, 1) = 1.;
    Ax(6, 2) = 1.;
    Ay(6, 1) = -1.;
};

inline constexpr auto adiabatic_bc_2D = [](const auto& in, auto& out) {
    const auto& [vals, ders, point, normal] = in;
    auto& [operators, rhs]                  = out;
    auto& [A0, A1, A2]                      = operators;

    // q * n = 0
    A0(0, 1) = normal[0];
    A0(0, 2) = normal[1];
};

#endif // L3STER_TESTS_KERNELS_HPP
