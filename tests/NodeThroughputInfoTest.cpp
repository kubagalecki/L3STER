#include "l3ster/comm/GatherNodeThroughputs.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include <cmath>
#include <iostream>

using namespace lstr;

int main(int argc, char* argv[])
{
    using namespace lstr;
    L3sterScopeGuard scope_guard{argc, argv};
    MpiComm          comm{};

    try
    {
        const auto throughputs = gatherNodeThroughputs(comm);

        if (comm.getRank() == 0)
        {
            if (throughputs.size() != static_cast< size_t >(comm.getSize()))
                throw std::logic_error{"Node throughputs should be calculated for every rank"};

            if (std::fabs(std::accumulate(begin(throughputs), end(throughputs), 0.) - 1.0) > 1e-15)
                throw std::logic_error{"Node throughputs should be normalized"};

            const auto zero_tp = std::ranges::find_if(throughputs, [](val_t tp) { return std::fabs(tp) < 1e-15; });
            if (zero_tp != end(throughputs))
                throw std::logic_error{"Node throughput calculation failed for at least one rank"};
        }
        else
        {
            if (!throughputs.empty())
                throw std::logic_error{"Node throughputs should only be gathered at rank 0"};
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Test failed: " << e.what();
        return EXIT_FAILURE;
    }
}