#include "Kernels.hpp"

template < el_o_t EO, q_o_t QO, KernelParams params, typename Kernel >
static void localEvalBenchImpl(benchmark::State& state, const Kernel& ker)
{
    constexpr auto ET = mesh::ElementType::Hex;
    constexpr auto QT = quad::QuadratureType::GaussLegendre;
    constexpr auto BT = basis::BasisType::Lagrange;

    const auto element   = getExampleHexElement< EO >();
    const auto node_vals = Eigen::Matrix< val_t, element.n_nodes, params.n_fields >::Random().eval();
    const auto quad_view = basis::getQuadratureView< BT, ET, EO, QT, QO >();
    const auto mapping   = map::TabulatedDomainMapping{quad_view.bases, std::span{element.data.vertices}};
    const auto fields    = map::FieldValuesAtPoints{*mapping.basis_values, mapping.physical_derivatives, node_vals};
    const auto kernel    = wrapDomainEquationKernel< params >(ker);
    const auto x         = Eigen::Vector< val_t, element.n_nodes * params.n_unknowns >::Random().eval();

    for (auto _ : state)
    {
        auto y = algsys::evaluateLocalOperator(kernel, mapping, fields, quad_view.weights, 0., x);
        benchmark::DoNotOptimize(y);
    }

    constexpr auto loc_mat_rows = element.n_nodes * params.n_unknowns;
    constexpr auto num_qps      = std::decay_t< decltype(*quad_view.bases.geom_basis) >::size;
    const auto     flops_per_qp =
        /* fill H */ params.n_equations * loc_mat_rows * 7 +
        /* operator evaluation */ params.n_equations * (4 * loc_mat_rows + 2);
    state.counters["DPFlops"] = benchmark::Counter{static_cast< double >(state.iterations()) * num_qps * flops_per_qp,
                                                   benchmark::Counter::kIsRate,
                                                   benchmark::Counter::kIs1000};
}

template < el_o_t EO >
static void BM_NS3DLocalEvaluation(benchmark::State& state)
{
    constexpr auto params = KernelParams{.dimension = 3, .n_equations = 8, .n_unknowns = 7, .n_fields = 7};
    localEvalBenchImpl< EO, 4 * EO - 1, params >(state, ns3d_kernel);
}

template < el_o_t EO >
static void BM_DiffS3DLocalEvaluation(benchmark::State& state)
{
    constexpr auto params = KernelParams{.dimension = 3, .n_equations = 8, .n_unknowns = 7, .n_fields = 7};
    localEvalBenchImpl< EO, 2 * EO, params >(state, diff3d_kernel);
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