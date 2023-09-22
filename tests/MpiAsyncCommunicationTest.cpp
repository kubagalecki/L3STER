#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include <algorithm>
#include <iostream>
#include <span>

template < typename T >
auto asRange(T& t)
{
    return std::span{std::addressof(t), 1};
}

int main(int argc, char* argv[])
{
    using namespace lstr;
    L3sterScopeGuard scope_guard{argc, argv};
    MpiComm          comm{MPI_COMM_WORLD};

    try
    {
        const auto size = comm.getSize();
        const auto rank = comm.getRank();

        if (size < 2)
            return 0;

        if (rank == 0)
        {
            auto in_msg = std::vector< char >(size);
            {
                std::vector< MpiComm::Request > rcv_requests;
                rcv_requests.reserve(static_cast< size_t >(size - 1));
                char message = 'z';
                for (int src = 1; src < size; ++src)
                    rcv_requests.push_back(comm.receiveAsync(asRange(in_msg[src]), src));
                std::ranges::for_each(rcv_requests, [](auto& req) {
                    if (req.test())
                        throw std::logic_error{"Message received too soon"};
                });
                for (int dest = 1; dest < size; ++dest)
                    comm.sendAsync(asRange(message), dest, 0).wait();
            }
            std::for_each(in_msg.cbegin() + 1, in_msg.cend(), [](char in) {
                if (in != 'a')
                    throw std::logic_error{"Message corrupted in transit"};
            });
        }
        else
        {
            char msg_in{}, msg_out = 'a';
            comm.receiveAsync(asRange(msg_in), 0).wait();
            if (msg_in != 'z')
                throw std::logic_error{"Message corrupted in transit"};
            comm.sendAsync(asRange(msg_out), 0, 0).wait();
        }
        auto request_to_cancel = comm.sendAsync(asRange(argc), 0, 0);
        request_to_cancel.cancel();
        request_to_cancel.wait();

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what();
        comm.abort();
        return EXIT_FAILURE;
    }
}
