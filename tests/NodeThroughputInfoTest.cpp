#include "l3ster/comm/GatherNodeThroughputs.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

#include <cmath>
#include <iostream>

using namespace lstr;

int main(int argc, char* argv[])
{
    using namespace lstr;
    L3sterScopeGuard scope_guard{argc, argv};
    MpiComm          comm{MPI_COMM_WORLD};

    const auto throughputs = gatherNodeThroughputs(comm);
    if (comm.getRank() == 0)
    {
        REQUIRE(throughputs.size() == static_cast< size_t >(comm.getSize()));
        REQUIRE(std::ranges::count_if(throughputs, [](val_t tp) { return std::fabs(tp) < 1e-15; }) == 0);
        REQUIRE(std::fabs(std::reduce(begin(throughputs), end(throughputs)) - 1.0) < 1e-15);
    }
    else
        REQUIRE(throughputs.empty());

    return EXIT_SUCCESS;
}