#ifndef L3STER_COMM_GATHERNODETHROUGHPUTS_HPP
#define L3STER_COMM_GATHERNODETHROUGHPUTS_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/defs/Typedefs.h"

#include <thread>

namespace lstr
{
inline std::vector< val_t > gatherNodeThroughputs(const MpiComm& comm)
{
    const auto     n_nodes       = comm.getSize();
    constexpr int  root          = 0;
    const unsigned my_throughput = std::thread::hardware_concurrency(); // TODO: scale this by clock frequency
    if (comm.getRank() == root)
    {
        std::vector< unsigned > throughputs(n_nodes);
        comm.gather(&my_throughput, throughputs.data(), 1, root);
        const auto throughput_sum = std::reduce(begin(throughputs), end(throughputs));
        if (throughput_sum == 0u)
            throw std::runtime_error{"Throughput computation failed on all nodes"};

        std::vector< val_t > ret_val{};
        ret_val.reserve(throughputs.size());
        std::ranges::transform(throughputs, std::back_inserter(ret_val), [&](auto tp) {
            return static_cast< val_t >(tp) / static_cast< val_t >(throughput_sum);
        });
        return ret_val;
    }
    else
    {
        comm.gather(&my_throughput, static_cast< unsigned* >(nullptr), 1, root);
        return {};
    }
}
} // namespace lstr
#endif // L3STER_COMM_GATHERNODETHROUGHPUTS_HPP
