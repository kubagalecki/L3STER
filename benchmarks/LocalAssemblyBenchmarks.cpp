#include "Common.hpp"

template < q_o_t QO, el_o_t EO >
static void BM_PhysBasisDersComputation(benchmark::State& state)
{
    constexpr auto QT        = QuadratureTypes::GLeg;
    constexpr auto BT        = BasisTypes::Lagrange;
    const auto     element   = getExampleHexElement< EO >();
    const auto     ref_basis = getReferenceBasisAtDomainQuadrature< BT, ElementTypes::Hex, EO, QT, QO >();
    const auto     jacobians = computeJacobiansAtQpoints(element, ref_basis.quadrature);
    for (auto _ : state)
    {
        const auto ders = computePhysBasisDersAtQpoints(ref_basis.basis_ders, jacobians);
        benchmark::DoNotOptimize(ders);
    }
}
#define DEF_PHYS_BAS_BENCH(QO, EO)                                                                                     \
    BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, QO, EO)                                                            \
        ->Name("Phys. basis der. at QPs computation [Hex, EO " #EO ", QO " #QO "]")                                    \
        ->Unit(benchmark::kMicrosecond);
#define DEF_PHYS_BAS_BENCH_SUITE(EO)                                                                                   \
    DEF_PHYS_BAS_BENCH(0, EO);                                                                                         \
    DEF_PHYS_BAS_BENCH(2, EO);                                                                                         \
    DEF_PHYS_BAS_BENCH(4, EO);                                                                                         \
    DEF_PHYS_BAS_BENCH(6, EO);                                                                                         \
    DEF_PHYS_BAS_BENCH(8, EO);                                                                                         \
    DEF_PHYS_BAS_BENCH(10, EO);                                                                                        \
    DEF_PHYS_BAS_BENCH(12, EO);                                                                                        \
    DEF_PHYS_BAS_BENCH(14, EO);
DEF_PHYS_BAS_BENCH_SUITE(1)
DEF_PHYS_BAS_BENCH_SUITE(2)
DEF_PHYS_BAS_BENCH_SUITE(3)
DEF_PHYS_BAS_BENCH_SUITE(4)

template < el_o_t EO >
static void BM_NS3DLocalAssembly(benchmark::State& state)
{
    constexpr auto  QT = QuadratureTypes::GLeg;
    constexpr auto  BT = BasisTypes::Lagrange;
    constexpr q_o_t QO = 4 * EO - 2;

    const auto  element         = getExampleHexElement< EO >();
    const auto& ref_bas_at_quad = getReferenceBasisAtDomainQuadrature< BT, ElementTypes::Hex, EO, QT, QO >();

    using nodal_vals_t            = Eigen::Matrix< val_t, element.n_nodes, 7 >;
    const nodal_vals_t nodal_vals = nodal_vals_t::Random();

    constexpr auto local_matrix_size = Element< ElementTypes ::Hex, EO >::n_nodes * 7;
    setMinStackSize(80'000'000 + 3 * sizeof(val_t) * local_matrix_size * local_matrix_size);

    constexpr auto ns3d_kernel = [](const auto& vals, const auto& ders, const auto&) noexcept {
        constexpr size_t nf = 7;
        constexpr size_t ne = 8;

        const auto& [u, v, w, p, ox, oy, oz]        = vals;
        const auto& [x_ders, y_ders, z_ders]        = ders;
        const auto& [ux, vx, wx, px, oxx, oyx, ozx] = x_ders;
        const auto& [uy, vy, wy, py, oxy, oyy, ozy] = y_ders;
        const auto& [uz, vz, wz, pz, oxz, oyz, ozz] = z_ders;

        using A_t   = Eigen::Matrix< val_t, ne, nf >;
        using F_t   = Eigen::Matrix< val_t, ne, 1 >;
        using ret_t = std::pair< std::array< A_t, 4 >, F_t >;
        ret_t ret_val;
        auto& [A0, A1, A2, A3] = ret_val.first;
        auto& F                = ret_val.second;
        for (auto& mat : ret_val.first)
            mat.setZero();
        F.setZero();

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

        F[0] = u * ux + v * uy + w * uz;
        F[1] = u * vx + v * vy + w * vz;
        F[2] = u * wx + v * wy + w * wz;

        return ret_val;
    };

    for (auto _ : state)
    {
        const auto local_system = assembleLocalSystem(ns3d_kernel, element, nodal_vals, ref_bas_at_quad, 0.);
        benchmark::DoNotOptimize(local_system);
    }
}
#define NS3D_ASSEMBLY_BENCH(ELO, UNIT)                                                                                 \
    BENCHMARK_TEMPLATE(BM_NS3DLocalAssembly, ELO)                                                                      \
        ->Name("Local NS3D system assembly [Hex, EO " #ELO "]")                                                        \
        ->Unit(benchmark::k##UNIT);
NS3D_ASSEMBLY_BENCH(1, Millisecond);
NS3D_ASSEMBLY_BENCH(2, Millisecond);
NS3D_ASSEMBLY_BENCH(3, Millisecond);
NS3D_ASSEMBLY_BENCH(4, Second);
NS3D_ASSEMBLY_BENCH(5, Second);
NS3D_ASSEMBLY_BENCH(6, Second);
