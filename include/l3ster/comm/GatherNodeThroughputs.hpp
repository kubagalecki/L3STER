#ifndef L3STER_COMM_GATHERNODETHROUGHPUTS_HPP
#define L3STER_COMM_GATHERNODETHROUGHPUTS_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/common/Typedefs.h"

#include "oneapi/tbb.h"

namespace lstr::comm
{
inline std::vector< val_t > gatherNodeThroughputs(const MpiComm& comm)
{
    const auto    n_nodes       = comm.getSize();
    constexpr int root          = 0;
    const size_t  my_throughput = oneapi::tbb::this_task_arena::max_concurrency(); // TODO: scale by clock frequency
    if (comm.getRank() == root)
    {
        std::vector< size_t > throughputs(n_nodes);
        comm.gather(std::views::single(my_throughput), throughputs.begin(), root);
        const auto           throughput_sum = std::reduce(begin(throughputs), end(throughputs));
        std::vector< val_t > retval{};
        retval.reserve(throughputs.size());
        std::ranges::transform(throughputs, std::back_inserter(retval), [&](auto tp) {
            return static_cast< val_t >(tp) / static_cast< val_t >(throughput_sum);
        });
        return retval;
    }
    else
    {
        comm.gather(std::views::single(my_throughput), static_cast< size_t* >(nullptr), root);
        return {};
    }
}
} // namespace lstr::comm
#endif // L3STER_COMM_GATHERNODETHROUGHPUTS_HPP
