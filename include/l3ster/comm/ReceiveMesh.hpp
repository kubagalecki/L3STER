#ifndef L3STER_COMM_RECEIVEMESH_HPP
#define L3STER_COMM_RECEIVEMESH_HPP

#include "l3ster/comm/SendMesh.hpp"

namespace lstr::comm
{
inline auto receivePartition(const MpiComm& comm, int source) -> mesh::SerializedPartition
{
    int msg_tag = 0;

    const auto n_doms  = std::invoke([&] {
        std::ranges::single_view< size_t > retval{};
        comm.receive(retval, source, msg_tag++);
        return retval.front();
    });
    const auto dom_ids = std::invoke([&] {
        std::vector< d_id_t > retval(n_doms);
        comm.receive(retval, source, msg_tag++);
        return retval;
    });
    const auto sizes   = std::invoke([&] {
        std::vector< size_t > retval(getSizeMsgLength(n_doms));
        comm.receive(retval, source, msg_tag++);
        return retval;
    });
    auto       size_it = sizes.begin();

    auto             retval              = mesh::SerializedPartition{};
    auto             messages            = std::vector< MpiComm::Request >{};
    constexpr size_t messages_per_domain = 6;
    messages.reserve(messages_per_domain * n_doms + 2);
    for (size_t i = 0; i < n_doms; ++i)
    {
        auto& domain = retval.domains.emplace(dom_ids[i], mesh::SerializedDomain{}).first->second;

        const auto node_msg_size = *size_it++;
        domain.element_nodes.resize(node_msg_size);
        messages.emplace_back(comm.receiveAsync(domain.element_nodes, source, msg_tag++));

        const auto data_msg_size = *size_it++;
        domain.element_data.resize(data_msg_size);
        messages.emplace_back(comm.receiveAsync(domain.element_data, source, msg_tag++));

        const auto id_msg_size = *size_it++;
        domain.element_ids.resize(id_msg_size);
        messages.emplace_back(comm.receiveAsync(domain.element_ids, source, msg_tag++));

        const auto offset_msg_size = *size_it++;
        domain.type_order_offsets.resize(offset_msg_size);
        domain.types.resize(offset_msg_size);
        domain.orders.resize(offset_msg_size);
        messages.emplace_back(comm.receiveAsync(domain.type_order_offsets, source, msg_tag++));
        messages.emplace_back(comm.receiveAsync(domain.types, source, msg_tag++));
        messages.emplace_back(comm.receiveAsync(domain.orders, source, msg_tag++));
    }

    const auto nodes_size = *size_it++;
    retval.n_owned_nodes  = *size_it++;
    retval.nodes          = util::ArrayOwner< n_id_t >(nodes_size);
    messages.emplace_back(comm.receiveAsync(retval.nodes, source, msg_tag++));

    const auto n_boundaries = *size_it++;
    retval.boundaries       = util::ArrayOwner< d_id_t >(n_boundaries);
    messages.emplace_back(comm.receiveAsync(retval.boundaries, source, msg_tag++));

    return retval;
}
} // namespace lstr::comm
#endif // L3STER_COMM_RECEIVEMESH_HPP
