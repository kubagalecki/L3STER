#include "Common.hpp"

template < q_o_t QO, el_o_t EO >
void BM_PhysBasisDersComputation(benchmark::State& state)
{
    constexpr auto QT      = QuadratureTypes::GLeg;
    constexpr auto BT      = BasisTypes::Lagrange;
    const auto     element = getExampleHexElement< EO >();
    for (auto _ : state)
    {
        const auto ders = computePhysicalBasesAtQpoints< QT, QO, BT >(element);
        benchmark::DoNotOptimize(ders);
    }
}
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 0, 1)
    ->Name("Phys. basis der. at QPs computation [Hex, EO1, QO0]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 2, 1)
    ->Name("Phys. basis der. at QPs computation [Hex, EO1, QO2]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 4, 1)
    ->Name("Phys. basis der. at QPs computation [Hex, EO1, QO4]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 6, 1)
    ->Name("Phys. basis der. at QPs computation [Hex, EO1, QO6]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 8, 1)
    ->Name("Phys. basis der. at QPs computation [Hex, EO1, QO8]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 10, 1)
    ->Name("Phys. basis der. at QPs computation [Hex, EO1, QO10]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 12, 1)
    ->Name("Phys. basis der. at QPs computation [Hex, EO1, QO12]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 14, 1)
    ->Name("Phys. basis der. at QPs computation [Hex, EO1, QO14]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 0, 2)
    ->Name("Phys. basis der. at QPs computation [Hex, EO2, QO0]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 2, 2)
    ->Name("Phys. basis der. at QPs computation [Hex, EO2, QO2]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 4, 2)
    ->Name("Phys. basis der. at QPs computation [Hex, EO2, QO4]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 6, 2)
    ->Name("Phys. basis der. at QPs computation [Hex, EO2, QO6]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 8, 2)
    ->Name("Phys. basis der. at QPs computation [Hex, EO2, QO8]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 10, 2)
    ->Name("Phys. basis der. at QPs computation [Hex, EO2, QO10]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 12, 2)
    ->Name("Phys. basis der. at QPs computation [Hex, EO2, QO12]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 14, 2)
    ->Name("Phys. basis der. at QPs computation [Hex, EO2, QO14]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 0, 3)
    ->Name("Phys. basis der. at QPs computation [Hex, EO3, QO0]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 2, 3)
    ->Name("Phys. basis der. at QPs computation [Hex, EO3, QO2]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 4, 3)
    ->Name("Phys. basis der. at QPs computation [Hex, EO3, QO4]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 6, 3)
    ->Name("Phys. basis der. at QPs computation [Hex, EO3, QO6]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 8, 3)
    ->Name("Phys. basis der. at QPs computation [Hex, EO3, QO8]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 10, 3)
    ->Name("Phys. basis der. at QPs computation [Hex, EO3, QO10]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 12, 3)
    ->Name("Phys. basis der. at QPs computation [Hex, EO3, QO12]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 14, 3)
    ->Name("Phys. basis der. at QPs computation [Hex, EO3, QO14]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 0, 4)
    ->Name("Phys. basis der. at QPs computation [Hex, EO4, QO0]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 2, 4)
    ->Name("Phys. basis der. at QPs computation [Hex, EO4, QO2]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 4, 4)
    ->Name("Phys. basis der. at QPs computation [Hex, EO4, QO4]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 6, 4)
    ->Name("Phys. basis der. at QPs computation [Hex, EO4, QO6]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 8, 4)
    ->Name("Phys. basis der. at QPs computation [Hex, EO4, QO8]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 10, 4)
    ->Name("Phys. basis der. at QPs computation [Hex, EO4, QO10]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 12, 4)
    ->Name("Phys. basis der. at QPs computation [Hex, EO4, QO12]")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 14, 4)
    ->Name("Phys. basis der. at QPs computation [Hex, EO4, QO14]")
    ->Unit(benchmark::kMicrosecond);

template < el_o_t EO >
void BM_NS3DLocalAssembly(benchmark::State& state)
{
    constexpr auto  QT = QuadratureTypes::GLeg;
    constexpr auto  BT = BasisTypes::Lagrange;
    constexpr q_o_t QO = 4 * EO - 2;

    const auto element = getExampleHexElement< EO >();

    using nodal_vals_t            = Eigen::Matrix< val_t, element.n_nodes, 7 >;
    const nodal_vals_t nodal_vals = nodal_vals_t::Random();

    constexpr auto local_matrix_size = Element< ElementTypes ::Hex, EO >::n_nodes * 7;
    setMinStackSize(80'000'000 + 3 * sizeof(val_t) * local_matrix_size * local_matrix_size);

    constexpr auto ns3d_kernel = [](const auto& vals, const auto& ders, const auto&) noexcept {
        constexpr size_t nf = 7;
        constexpr size_t ne = 8;

        const auto& [u, v, w, p, ox, oy, oz] = vals;
        const auto& [ux, vx, wx]             = ders[0];
        const auto& [uy, vy, wy]             = ders[1];
        const auto& [uz, vz, wz]             = ders[2];

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
    const std::array der_indices{0, 1, 2};

    for (auto _ : state)
    {
        const auto local_system = assembleLocalSystem< QT, QO, BT >(ns3d_kernel, element, nodal_vals, der_indices);
        benchmark::DoNotOptimize(local_system);
    }
}
BENCHMARK_TEMPLATE(BM_NS3DLocalAssembly, 1)->Name("Local NS3D system assembly [EO1]")->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_NS3DLocalAssembly, 2)->Name("Local NS3D system assembly [EO2]")->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_NS3DLocalAssembly, 3)->Name("Local NS3D system assembly [EO3]")->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_NS3DLocalAssembly, 4)->Name("Local NS3D system assembly [EO4]")->Unit(benchmark::kSecond);
BENCHMARK_TEMPLATE(BM_NS3DLocalAssembly, 5)->Name("Local NS3D system assembly [EO5]")->Unit(benchmark::kSecond);
BENCHMARK_TEMPLATE(BM_NS3DLocalAssembly, 6)->Name("Local NS3D system assembly [EO6]")->Unit(benchmark::kSecond);
