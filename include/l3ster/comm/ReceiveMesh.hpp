#ifndef L3STER_COMM_RECEIVEMESH_HPP
#define L3STER_COMM_RECEIVEMESH_HPP

#include "SendMesh.hpp"

namespace lstr
{
inline SerializedPartition receivePartition(const MpiComm& comm, int source)
{
    int msg_tag = 0;

    const auto n_doms  = comm.template receive< size_t >(source, msg_tag++);
    const auto dom_ids = comm.template receive< d_id_t >(n_doms, source, msg_tag++);
    const auto sizes   = comm.template receive< size_t >(detail::getSizeMsgLength(n_doms), source, msg_tag++);

    SerializedPartition             retval{};
    std::vector< MpiComm::Request > messages{};
    constexpr size_t                messages_per_domain = 6;
    messages.reserve(messages_per_domain * n_doms + 2);
    for (size_t i = 0; i < n_doms; ++i)
    {
        auto& domain = retval.domains.emplace(dom_ids[i], SerializedDomain{}).first->second;

        const auto node_msg_size = sizes[i * 4];
        domain.element_nodes.resize(node_msg_size);
        messages.emplace_back(comm.receiveAsync(domain.element_nodes, source, msg_tag++));

        const auto data_msg_size = sizes[i * 4 + 1];
        domain.element_data.resize(data_msg_size);
        messages.emplace_back(comm.receiveAsync(domain.element_data, source, msg_tag++));

        const auto id_msg_size = sizes[i * 4 + 2];
        domain.element_ids.resize(id_msg_size);
        messages.emplace_back(comm.receiveAsync(domain.element_ids, source, msg_tag++));

        const auto offset_msg_size = sizes[i * 4 + 3];
        domain.type_order_offsets.resize(offset_msg_size);
        domain.types.resize(offset_msg_size);
        domain.orders.resize(offset_msg_size);
        messages.emplace_back(comm.receiveAsync(domain.type_order_offsets, source, msg_tag++));
        messages.emplace_back(comm.receiveAsync(domain.types, source, msg_tag++));
        messages.emplace_back(comm.receiveAsync(domain.orders, source, msg_tag++));
    }

    const auto nodes_size = sizes[4 * n_doms];
    retval.nodes.resize(nodes_size);
    messages.emplace_back(comm.receiveAsync(retval.nodes, source, msg_tag++));

    const auto ghost_nodes_size = sizes.back();
    retval.ghost_nodes.resize(ghost_nodes_size);
    messages.emplace_back(comm.receiveAsync(retval.ghost_nodes, source, msg_tag));

    return retval;
}
} // namespace lstr
#endif // L3STER_COMM_RECEIVEMESH_HPP
