#include "Kernels.hpp"

template < el_o_t EO >
static void BM_NS3DLocalEvaluation(benchmark::State& state)
{
    constexpr auto  QT = quad::QuadratureType::GaussLegendre;
    constexpr auto  BT = basis::BasisType::Lagrange;
    constexpr q_o_t QO = 4 * EO - 1;

    const auto element = getExampleHexElement< EO >();
    auto       dom_map = typename mesh::MeshPartition< EO >::domain_map_t{};
    mesh::pushToDomain(dom_map[0], element);
    const auto mesh          = mesh::MeshPartition< EO >{dom_map, 0, element.nodes.size(), {}};
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
                              /* fill H */ params.n_equations * n_nodes * params.n_unknowns * 7 +
                              /* operator evaluation */ params.n_equations * (n_nodes * params.n_unknowns * 4 + 2);
    const auto n_qp           = ref_bas_at_quad.quadrature.size;
    state.counters["DPFlops"] = benchmark::Counter{static_cast< double >(state.iterations()) * n_qp * flops_per_qp,
                                                   benchmark::Counter::kIsRate,
                                                   benchmark::Counter::kIs1000};
}

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
    const auto mesh               = mesh::MeshPartition< EO >{dom_map, 0, element.nodes.size(), {}};
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
                              /* fill H */ params.n_equations * n_nodes * params.n_unknowns * 7 +
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
NS3D_EVAL_BENCH(6, Millisecond);

#define DIFF3D_EVAL_BENCH(ELO, UNIT)                                                                                   \
    BENCHMARK_TEMPLATE(BM_DiffS3DLocalEvaluation, ELO)                                                                 \
        ->Name("Local Diff3D operator evaluation [Hex, EO " #ELO "]")                                                  \
        ->Unit(benchmark::k##UNIT);
DIFF3D_EVAL_BENCH(2, Microsecond);
DIFF3D_EVAL_BENCH(4, Microsecond);
DIFF3D_EVAL_BENCH(6, Millisecond);