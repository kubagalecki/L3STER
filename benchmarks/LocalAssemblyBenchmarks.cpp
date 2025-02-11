#include "Common.hpp"

template < q_o_t QO, el_o_t EO >
static void BM_PhysBasisDersComputation(benchmark::State& state)
{
    constexpr auto QT          = quad::QuadratureType::GaussLegendre;
    constexpr auto BT          = basis::BasisType::Lagrange;
    const auto     element     = getExampleHexElement< EO >();
    const auto&    ref_basis   = basis::getReferenceBasisAtDomainQuadrature< BT, mesh::ElementType::Hex, EO, QT, QO >();
    const auto     jac_mat_gen = map::getNatJacobiMatGenerator(element.getData());
    for (auto _ : state)
        for (size_t qp_ind = 0; qp_ind < ref_basis.quadrature.size; ++qp_ind)
        {
            const auto  jac_mat  = jac_mat_gen(ref_basis.quadrature.points[qp_ind]);
            const auto& ref_ders = ref_basis.basis.derivatives[qp_ind];
            benchmark::DoNotOptimize(map::computePhysBasisDers(jac_mat, ref_ders));
        }

    constexpr auto n_nodes = mesh::Element< mesh::ElementType::Hex, EO >::n_nodes;
    const auto     n_qp    = ref_basis.quadrature.size;
    state.counters["DPFlops"] =
        benchmark::Counter{static_cast< double >(state.iterations()) * 3 * 3 * 2 * n_nodes * n_qp,
                           benchmark::Counter::kIsRate,
                           benchmark::Counter::kIs1000};
}
#define DEF_PHYS_BAS_BENCH(QO, EO)                                                                                     \
    BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, QO, EO)                                                            \
        ->Name("Phys. basis der. at QPs computation [Hex, EO " #EO ", QO " #QO "]")                                    \
        ->Unit(benchmark::kMicrosecond);
#define DEF_PHYS_BAS_BENCH_SUITE(EO)                                                                                   \
    DEF_PHYS_BAS_BENCH(0, EO);                                                                                         \
    DEF_PHYS_BAS_BENCH(6, EO);                                                                                         \
    DEF_PHYS_BAS_BENCH(18, EO);

DEF_PHYS_BAS_BENCH_SUITE(1)
DEF_PHYS_BAS_BENCH_SUITE(2)
DEF_PHYS_BAS_BENCH_SUITE(4)
DEF_PHYS_BAS_BENCH_SUITE(6)

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

template < el_o_t EO >
static void BM_NS3DLocalAssembly(benchmark::State& state)
{
    constexpr auto  QT = quad::QuadratureType::GaussLegendre;
    constexpr auto  BT = basis::BasisType::Lagrange;
    constexpr q_o_t QO = 4 * EO - 1;

    const auto element = getExampleHexElement< EO >();

    constexpr size_t n_fields     = 7;
    constexpr size_t n_eq         = 8;
    using nodal_vals_t            = Eigen::Matrix< val_t, element.n_nodes, n_fields >;
    const nodal_vals_t nodal_vals = nodal_vals_t::Random();

    constexpr auto n_nodes      = mesh::Element< mesh::ElementType ::Hex, EO >::n_nodes;
    constexpr auto loc_mat_rows = n_nodes * n_fields;

    const auto& ref_bas_at_quad =
        basis::getReferenceBasisAtDomainQuadrature< BT, mesh::ElementType::Hex, EO, QT, QO >();

    constexpr auto params = KernelParams{.dimension = 3, .n_equations = 8, .n_unknowns = 7, .n_fields = 7};
    constexpr auto kernel = wrapDomainEquationKernel< params >(ns3d_kernel);

    for (auto _ : state)
    {
        const auto& local_sys = algsys::assembleLocalSystem(kernel, element, nodal_vals, ref_bas_at_quad, 0.);
        benchmark::DoNotOptimize(&local_sys);
        benchmark::ClobberMemory();
    }

    const auto flops_per_qp = /* physical basis derivative computation */ n_nodes * 3 * 3 * 2 +
                              /* field value computation */ n_fields * n_nodes * 2 * 4 +
                              /* rank update matrix creation */ loc_mat_rows * n_eq * 7 +
                              /* rank update flops */ (loc_mat_rows + 1) * (loc_mat_rows + 1) / 2 * (2 * n_eq + 1);
    const auto n_qp           = ref_bas_at_quad.quadrature.size;
    state.counters["DPFlops"] = benchmark::Counter{static_cast< double >(state.iterations()) * n_qp * flops_per_qp,
                                                   benchmark::Counter::kIsRate,
                                                   benchmark::Counter::kIs1000};
}
#define NS3D_ASSEMBLY_BENCH(ELO, UNIT)                                                                                 \
    BENCHMARK_TEMPLATE(BM_NS3DLocalAssembly, ELO)                                                                      \
        ->Name("Local NS3D system assembly [Hex, EO " #ELO "]")                                                        \
        ->Unit(benchmark::k##UNIT);
NS3D_ASSEMBLY_BENCH(2, Microsecond);
NS3D_ASSEMBLY_BENCH(4, Millisecond);
NS3D_ASSEMBLY_BENCH(6, Second);

template < el_o_t EO >
static void BM_NS3DLocalEvaluation(benchmark::State& state)
{
    constexpr auto  QT = quad::QuadratureType::GaussLegendre;
    constexpr auto  BT = basis::BasisType::Lagrange;
    constexpr q_o_t QO = 4 * EO - 1;

    const auto element = getExampleHexElement< EO >();
    auto       dom_map = typename mesh::MeshPartition< EO >::domain_map_t{};
    mesh::pushToDomain(dom_map[0], element);
    const auto mesh          = mesh::MeshPartition< EO >{dom_map, 0, element.getNodes().size(), {}};
    const auto local_element = mesh::LocalElementView{element, mesh, {}};

    constexpr size_t n_fields     = 7;
    constexpr size_t n_eq         = 8;
    using nodal_vals_t            = Eigen::Matrix< val_t, element.n_nodes, n_fields >;
    const nodal_vals_t nodal_vals = nodal_vals_t::Random();

    constexpr auto n_nodes      = mesh::Element< mesh::ElementType ::Hex, EO >::n_nodes;
    constexpr auto loc_mat_rows = n_nodes * n_fields;

    const auto& ref_bas_at_quad =
        basis::getReferenceBasisAtDomainQuadrature< BT, mesh::ElementType::Hex, EO, QT, QO >();

    constexpr auto params = KernelParams{.dimension = 3, .n_equations = 8, .n_unknowns = 7, .n_fields = 7};
    constexpr auto kernel = wrapDomainEquationKernel< params >(ns3d_kernel);

    Eigen::Vector< val_t, element.n_nodes * params.n_unknowns > x;
    x.setRandom();

    for (auto _ : state)
    {
        auto y = algsys::evaluateLocalOperator(kernel, local_element, nodal_vals, ref_bas_at_quad, 0., x);
        benchmark::DoNotOptimize(y);
    }

    const auto flops_per_qp = /* physical basis derivative computation */ n_nodes * 3 * 3 * 2 +
                              /* field value computation */ n_fields * n_nodes * 2 * 4 +
                              /* operator evaluation */ params.n_equations * (n_nodes * params.n_unknowns * 4 + 2);
    const auto n_qp           = ref_bas_at_quad.quadrature.size;
    state.counters["DPFlops"] = benchmark::Counter{static_cast< double >(state.iterations()) * n_qp * flops_per_qp,
                                                   benchmark::Counter::kIsRate,
                                                   benchmark::Counter::kIs1000};
}
#define NS3D_EVAL_BENCH(ELO, UNIT)                                                                                     \
    BENCHMARK_TEMPLATE(BM_NS3DLocalEvaluation, ELO)                                                                    \
        ->Name("Local NS3D operator evaluation [Hex, EO " #ELO "]")                                                    \
        ->Unit(benchmark::k##UNIT);
NS3D_EVAL_BENCH(2, Microsecond);
NS3D_EVAL_BENCH(4, Millisecond);
NS3D_EVAL_BENCH(6, Second);

template < el_o_t EO >
static void BM_DiffS3DLocalEvaluation(benchmark::State& state)
{
    constexpr auto  params = KernelParams{.dimension = 3, .n_equations = 7, .n_unknowns = 4};
    constexpr auto  QT     = quad::QuadratureType::GaussLegendre;
    constexpr auto  BT     = basis::BasisType::Lagrange;
    constexpr q_o_t QO     = 2 * EO;

    const auto element = getExampleHexElement< EO >();
    auto       dom_map = typename mesh::MeshPartition< EO >::domain_map_t{};
    mesh::pushToDomain(dom_map[0], element);
    const auto mesh               = mesh::MeshPartition< EO >{dom_map, 0, element.getNodes().size(), {}};
    const auto local_element      = mesh::LocalElementView{element, mesh, {}};
    using nodal_vals_t            = Eigen::Matrix< val_t, element.n_nodes, params.n_fields >;
    const nodal_vals_t nodal_vals = nodal_vals_t::Random();
    constexpr auto     n_nodes    = mesh::Element< mesh::ElementType ::Hex, EO >::n_nodes;

    const auto& ref_bas_at_quad =
        basis::getReferenceBasisAtDomainQuadrature< BT, mesh::ElementType::Hex, EO, QT, QO >();

    constexpr auto kernel = wrapDomainEquationKernel< params >(diff3d_kernel);

    Eigen::Vector< val_t, element.n_nodes * params.n_unknowns > x;
    x.setRandom();

    for (auto _ : state)
    {
        auto y = algsys::evaluateLocalOperator(kernel, local_element, nodal_vals, ref_bas_at_quad, 0., x);
        benchmark::DoNotOptimize(y);
    }

    const auto flops_per_qp = /* physical basis derivative computation */ n_nodes * 3 * 3 * 2 +
                              /* field value computation */ params.n_fields * n_nodes * 2 * 4 +
                              /* operator evaluation */ params.n_equations * (n_nodes * params.n_unknowns * 4 + 2);
    const auto n_qp           = ref_bas_at_quad.quadrature.size;
    state.counters["DPFlops"] = benchmark::Counter{static_cast< double >(state.iterations()) * n_qp * flops_per_qp,
                                                   benchmark::Counter::kIsRate,
                                                   benchmark::Counter::kIs1000};
}
#define DIFF3D_EVAL_BENCH(ELO, UNIT)                                                                                   \
    BENCHMARK_TEMPLATE(BM_DiffS3DLocalEvaluation, ELO)                                                                 \
        ->Name("Local Diff3D operator evaluation [Hex, EO " #ELO "]")                                                  \
        ->Unit(benchmark::k##UNIT);
DIFF3D_EVAL_BENCH(2, Microsecond);
DIFF3D_EVAL_BENCH(4, Microsecond);
DIFF3D_EVAL_BENCH(6, Millisecond);