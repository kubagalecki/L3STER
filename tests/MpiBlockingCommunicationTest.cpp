#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include <iostream>

int main(int argc, char* argv[])
{
    using namespace lstr;
    L3sterScopeGuard scope_guard{argc, argv};
    MpiComm          comm{};
    try
    {
        const auto size = comm.getSize();
        const auto rank = comm.getRank();

        if (size < 2)
            return 0;

        const std::vector expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

        if (rank != 0)
        {
            auto received = expected;
            comm.receive(received, rank - 1);
            if (received != expected)
                throw std::runtime_error{"message corrupted in transit"};
        }

        if (rank != size - 1)
            comm.send(expected, rank + 1);

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what();
        comm.abort();
        return EXIT_FAILURE;
    }
}
