#include "Kernels.hpp"

template < LocalEvalStrategy LES, KernelParams params, mesh::ElementType ET, el_o_t EO >
static auto evalDiffusionOperatorSumFact(const mesh::LocalElementView< ET, EO >&  element,
                                         const algsys::Operand< ET, EO, params >& x)
{
    constexpr auto                    kernel  = std::invoke([] {
        if constexpr (ET == mesh::ElementType::Quad)
            return wrapDomainEquationKernel< params >(diff2d_kernel);
        else
            return wrapDomainEquationKernel< params >(diff3d_kernel);
    });
    constexpr auto                    n_nodes = mesh::Element< ET, EO >::n_nodes;
    constexpr Eigen::Index            nukn    = params.n_unknowns;
    algsys::Operand< ET, EO, params > y;
    const auto                        x_fill = [&x](std::span< val_t > to_fill) {
        using map_t = Eigen::Map< Eigen::Matrix< val_t, n_nodes, params.n_unknowns * params.n_rhs > >;
        auto to_fill_map = map_t{to_fill.data()};
        for (Eigen::Index i = 0; i != n_nodes; ++i)
            to_fill_map.row(i) = x.template middleRows< nukn >(i * nukn).reshaped().transpose();
    };
    const auto y_fill = [&y](std::span< val_t > result) {
        using map_t = Eigen::Map< util::eigen::RowMajorMatrix< val_t, n_nodes, params.n_unknowns * params.n_rhs > >;
        const auto result_map = map_t{result.data()};
        for (Eigen::Index i = 0; i != n_nodes; ++i)
            y.template middleRows< nukn >(i * nukn).reshaped().transpose() = result_map.row(i);
    };
    constexpr auto asm_opts        = AssemblyOptions{.value_order = 2, .eval_strategy = LES};
    constexpr auto asm_opts_ctwrpr = util::ConstexprValue< asm_opts >{};
    algsys::evalLocalOperatorSumFact(kernel, element, x_fill, {}, y_fill, asm_opts_ctwrpr);
    return y;
}

template < el_o_t EO, LocalEvalStrategy LES >
static void BM_SumFactQuadDiff(benchmark::State& state)
{
    constexpr auto ET           = mesh::ElementType::Quad;
    auto           el_nodes     = typename mesh::Element< ET, EO >::node_array_t{};
    constexpr auto invalid_node = std::numeric_limits< n_id_t >::max();
    el_nodes.fill(invalid_node);
    n_id_t n = 0;
    for (const auto& bnd_node_ind : mesh::ElementTraits< mesh::Element< ET, EO > >::boundary_node_inds)
        el_nodes[bnd_node_ind] = n++;
    for (const auto& int_node_ind : mesh::ElementTraits< mesh::Element< ET, EO > >::internal_node_inds)
        el_nodes[int_node_ind] = n++;
    const auto data =
        mesh::ElementData< ET, EO >{{Point{1., 1., 0.}, Point{2., 1., 0.}, Point{1., 3., 0.}, Point{3., 4., 0.}}};
    const auto element = mesh::Element{el_nodes, data, 0};
    auto       dom_map = typename mesh::MeshPartition< EO >::domain_map_t{};
    mesh::pushToDomain(dom_map[0], element);
    const auto global_mesh   = mesh::MeshPartition< EO >{std::move(dom_map), {}};
    const auto local_element = mesh::LocalElementView{element, global_mesh, {}};

    constexpr auto params = KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = 3};
    const auto     x      = algsys::Operand< ET, EO, params >::Random().eval();
    auto           y      = algsys::Operand< ET, EO, params >{};

    for (auto _ : state)
    {
        y = evalDiffusionOperatorSumFact< LES, params >(local_element, x);
        benchmark::DoNotOptimize(y);
    }
}

#define DIFF2D_SF_BENCH(ELO, UNIT)                                                                                     \
    BENCHMARK(BM_SumFactQuadDiff< ELO, LocalEvalStrategy::SumFactorization >)                                          \
        ->Name("Diff2D sum-fact, EO=" #ELO)                                                                            \
        ->Unit(benchmark::k##UNIT);                                                                                    \
    BENCHMARK(BM_SumFactQuadDiff< ELO, LocalEvalStrategy::SumFactorizationOddEvenDecomposition >)                      \
        ->Name("Diff2D sum-fact + odd-even, EO=" #ELO)                                                                 \
        ->Unit(benchmark::k##UNIT)

DIFF2D_SF_BENCH(2, Nanosecond);
DIFF2D_SF_BENCH(4, Microsecond);
DIFF2D_SF_BENCH(6, Microsecond);
DIFF2D_SF_BENCH(8, Microsecond);
