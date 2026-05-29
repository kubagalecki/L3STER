#include "l3ster/l3ster.hpp"

#define BENCHMARK_MAIN_NOT_DEFINED
#include <benchmark/benchmark.h>

int main(int argc, char** argv)
{
    const auto stack_guard = lstr::util::StackSizeGuard{1uz << 30};
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return EXIT_FAILURE;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}