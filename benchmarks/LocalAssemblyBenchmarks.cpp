#include "Kernels.hpp"

template < el_o_t EO >
static void BM_NS3DLocalAssembly(benchmark::State& state)
{
    constexpr auto  ET     = mesh::ElementType::Hex;
    constexpr auto  QT     = quad::QuadratureType::GaussLegendre;
    constexpr auto  BT     = basis::BasisType::Lagrange;
    constexpr q_o_t QO     = 4 * EO - 1;
    constexpr auto  params = KernelParams{.dimension = 3, .n_equations = 8, .n_unknowns = 7, .n_fields = 7};

    const auto     element   = getExampleHexElement< EO >();
    const auto     node_vals = Eigen::Matrix< val_t, element.n_nodes, params.n_fields >::Random().eval();
    const auto     quad_view = basis::getQuadratureView< BT, ET, EO, QT, QO >();
    const auto     mapping   = map::TabulatedDomainMapping{quad_view.bases, std::span{element.data.vertices}};
    const auto     fields    = map::FieldValuesAtPoints{*mapping.basis_values, mapping.physical_derivatives, node_vals};
    constexpr auto kernel    = wrapDomainEquationKernel< params >(ns3d_kernel);

    for (auto _ : state)
    {
        const auto& local_sys = algsys::assembleLocalSystem(kernel, mapping, fields, quad_view.weights, 0.);
        benchmark::DoNotOptimize(&local_sys);
        benchmark::ClobberMemory();
    }

    constexpr auto loc_mat_rows = element.n_nodes * params.n_unknowns;
    constexpr auto num_qps      = std::decay_t< decltype(*quad_view.bases.geom_basis) >::size;
    const auto     flops_per_qp =
        /* rank update matrix creation */ loc_mat_rows * params.n_equations * 7 +
        /* rank update flops */ (loc_mat_rows + 1) * (loc_mat_rows + 1) / 2 * (2 * params.n_equations + 1);

    state.counters["DPFlops"] = benchmark::Counter{static_cast< double >(state.iterations()) * num_qps * flops_per_qp,
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
