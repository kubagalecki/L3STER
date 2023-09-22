#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include <iostream>

#include "Common.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;
    const auto scope_guard = L3sterScopeGuard{argc, argv};
    MpiComm    comm{MPI_COMM_WORLD, MPI_ERRORS_RETURN};

    const auto size = comm.getSize();
    const auto rank = comm.getRank();

    if (size < 2)
        return EXIT_SUCCESS;

    const auto expected = std::vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    if (rank != 0)
    {
        auto received = expected;
        comm.receive(received, rank - 1);
        REQUIRE(received == expected);
    }

    if (rank != size - 1)
        comm.send(expected, rank + 1, 0);

    CHECK_THROWS(comm.send(std::views::single(0), size, 0));

    // Code coverage for the branch checking initialization/finalization
    const auto scope_guard2 = L3sterScopeGuard{argc, argv};
}
