#include "numa/MpiComm.hpp"

int main(int argc, char* argv[])
{
    std::pmr::unsynchronized_pool_resource resource;
    lstr::MpiComm                          comm{&resource, argc, argv};
}