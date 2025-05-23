#include "l3ster/l3ster.hpp"

int main(int argc, char* argv[])
{
    // This is the main L3STER scope guard which initalizes MPI, Kokkos, and possibly others
    // *** L3STER objects cannot outlive it ***
    const auto scope_guard = lstr::L3sterScopeGuard{argc, argv};

    // L3STER wrapper for an MPI communicator - you will need to wrap it in a std::shared_ptr in subsequent examples
    const auto comm = lstr::MpiComm{MPI_COMM_WORLD};

    // Print message
    std::println("Hello, World!\nLove,\nL3STER app, rank {} of {}\n", comm.getRank(), comm.getSize());
}