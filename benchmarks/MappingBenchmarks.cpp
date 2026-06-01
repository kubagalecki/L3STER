#include "Common.hpp"

inline constexpr auto BT = basis::BasisType::Lagrange;
inline constexpr auto EO = 1;
inline constexpr auto QT = quad::QuadratureType::GaussLegendre;
inline constexpr auto QO = 10;

template < el_o_t GO >
static void BM_MappingBenchmark(benchmark::State& state)
{
    constexpr auto element   = std::invoke([] {
        if constexpr (GO == 1)
            return getExampleHexElement< EO >();
        else
            return getExampleHex2Element< EO >();
    });
    constexpr auto ET        = std::decay_t< decltype(element) >::type;
    const auto     quad_view = basis::getQuadratureView< BT, ET, EO, QT, QO >();

    for (auto _ : state)
    {
        auto map = map::TabulatedDomainMapping{quad_view.bases, std::span{element.data.vertices}};
        benchmark::DoNotOptimize(map);
    }

    constexpr auto num_qp      = std::decay_t< decltype(*quad_view.bases.geom_basis) >::size;
    state.counters["Points/s"] = {
        static_cast< double >(state.iterations() * num_qp), benchmark::Counter::kIsRate, benchmark::Counter::kIs1000};
}
BENCHMARK(BM_MappingBenchmark< 1 >)->Name("Compute mapping [hex1]");
BENCHMARK(BM_MappingBenchmark< 2 >)->Name("Compute mapping [hex2]");
