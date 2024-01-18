#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include <iostream>

#include "Common.hpp"

struct DummyData
{
    std::array< int, 3 > i3;
    double               d1;
};
static constexpr bool operator==(const DummyData& a, const DummyData& b)
{
    return a.i3 == b.i3 and a.d1 == b.d1;
}

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

    // Test for non-built-in types
    const auto d1 = DummyData{1, 2, 3, 4.};
    const auto d2 = DummyData{5, 6, 7, 8.};
    if (rank == 0)
    {
        auto data = std::vector< DummyData >{d1, d2};
        comm.broadcast(data, 0);
        for (int dest_rank = 1; dest_rank != size; ++dest_rank)
            comm.send(std::as_const(data), dest_rank, 0);
    }
    else
    {
        auto data = std::vector< DummyData >(2);
        comm.broadcast(data, 0);
        REQUIRE(data.front() == d1);
        REQUIRE(data.back() == d2);
        data = std::vector< DummyData >(2);
        comm.receive(data, 0, 0);
        REQUIRE(data.front() == d1);
        REQUIRE(data.back() == d2);
    }

    CHECK_THROWS(comm.send(std::views::single(0), size, 0));

    // Code coverage for the branch checking initialization/finalization
    const auto scope_guard2 = L3sterScopeGuard{argc, argv};
}
