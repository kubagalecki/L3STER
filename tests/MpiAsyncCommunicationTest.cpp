#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/util/GlobalResource.hpp"

#include <algorithm>
#include <iostream>

int main(int argc, char* argv[])
{
    lstr::GlobalResource< lstr::MpiScopeGuard >::init(argc, argv);
    lstr::MpiComm comm{};
    try
    {
        const auto size = comm.getSize();
        const auto rank = comm.getRank();

        if (size < 2)
            return 0;

        if (rank == 0)
        {
            std::vector< char > in_msg(size);
            {
                std::vector< lstr::MpiComm::Request > rcv_requests;
                rcv_requests.reserve(size - 1);
                char message = 'z';
                for (int src = 1; src < size; ++src)
                    rcv_requests.push_back(comm.receiveAsync(&in_msg[src], 1, src));
                std::ranges::for_each(rcv_requests, [](auto& req) {
                    if (req.test())
                        throw std::logic_error{"Message received too soon"};
                });
                for (int dest = 1; dest < size; ++dest)
                    comm.sendAsync(&message, 1, dest).wait();
            }
            std::for_each(in_msg.cbegin() + 1, in_msg.cend(), [](char in) {
                if (in != 'a')
                    throw std::logic_error{"Message corrupted in transit"};
            });
        }
        else
        {
            char msg_in, msg_out = 'a';
            comm.receiveAsync(&msg_in, 1, 0).wait();
            if (msg_in != 'z')
                throw std::logic_error{"Message corrupted in transit"};
            comm.sendAsync(&msg_out, 1, 0).wait();
        }

        return EXIT_SUCCESS;
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << e.what();
        comm.abort();
        return EXIT_FAILURE;
    }
}
