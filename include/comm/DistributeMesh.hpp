#ifndef L3STER_COMM_DISTRIBUTEMESH_HPP
#define L3STER_COMM_DISTRIBUTEMESH_HPP

#include "comm/DeserializeMesh.hpp"
#include "comm/MpiComm.hpp"

namespace lstr
{
namespace detail
{
inline auto sendNDoms(const MpiComm& comm, const SerializedPartition& part, int destination, int tag)
{
    const size_t n_domains = part.domains.size();
    return comm.sendAsync(&n_domains, 1, destination, tag);
}

inline auto sendIds(const MpiComm& comm, const SerializedPartition& part, int destination, int tag)
{
    std::vector< d_id_t > dom_ids(part.domains.size());
    std::ranges::transform(part.domains, begin(dom_ids), [](const auto& pair) { return pair.first; });
    return comm.sendAsync(dom_ids, destination, tag);
}

inline size_t getSizeMsgLength(size_t n_domains)
{
    constexpr size_t n_domain_size_params = 4;
    return n_domain_size_params * n_domains + 2; // 4 per domain + 2 for (ghost) nodes
}

inline auto sendSizes(const MpiComm& comm, const SerializedPartition& part, int destination, int tag)
{
    const auto            msg_size = getSizeMsgLength(part.domains.size());
    std::vector< size_t > size_info{};
    size_info.reserve(msg_size);
    for (const auto& dom : part.domains)
    {
        size_info.push_back(dom.second.element_nodes.size());
        size_info.push_back(dom.second.element_data.size());
        size_info.push_back(dom.second.element_ids.size());
        size_info.push_back(dom.second.type_order_offsets.size());
    }
    size_info.push_back(part.nodes.size());
    size_info.push_back(part.ghost_nodes.size());
    return comm.sendAsync(size_info, destination, tag);
}
} // namespace detail

inline void sendPartition(const MpiComm& comm, const SerializedPartition& part, int destination)
{
    int msg_tag = 0;

    const size_t n_messages = 3 + 6 * part.domains.size() + 2; // 3 prelims, 6 per domain, 2 for (ghost)nodes
    std::vector< MpiComm::Request > messages;                  // communication completion ensured via RAII
    messages.reserve(n_messages);

    // prelims
    messages.emplace_back(detail::sendNDoms(comm, part, destination, msg_tag++));
    messages.emplace_back(detail::sendIds(comm, part, destination, msg_tag++));
    messages.emplace_back(detail::sendSizes(comm, part, destination, msg_tag++));

    // domain data
    for (const auto& [id, dom] : part.domains)
    {
        messages.emplace_back(comm.sendAsync(dom.element_nodes, destination, msg_tag++));
        messages.emplace_back(comm.sendAsync(dom.element_data, destination, msg_tag++));
        messages.emplace_back(comm.sendAsync(dom.element_ids, destination, msg_tag++));
        messages.emplace_back(comm.sendAsync(dom.type_order_offsets, destination, msg_tag++));
        messages.emplace_back(comm.sendAsync(dom.types, destination, msg_tag++));
        messages.emplace_back(comm.sendAsync(dom.orders, destination, msg_tag++));
    }

    // nodes
    messages.emplace_back(comm.sendAsync(part.nodes, destination, msg_tag++));
    messages.emplace_back(comm.sendAsync(part.ghost_nodes, destination, msg_tag));
}

inline SerializedPartition receivePartition(const MpiComm& comm, int source)
{
    int msg_tag = 0;

    const auto n_doms  = comm.template receive< size_t >(source, msg_tag++);
    const auto dom_ids = comm.template receive< d_id_t >(n_doms, source, msg_tag++);
    const auto sizes   = comm.template receive< size_t >(detail::getSizeMsgLength(n_doms), source, msg_tag++);

    SerializedPartition             ret_val{};
    std::vector< MpiComm::Request > messages{};
    constexpr size_t                messages_per_domain = 6;
    messages.reserve(messages_per_domain * n_doms + 2);
    for (size_t i = 0; i < n_doms; ++i)
    {
        auto& domain = ret_val.domains.emplace(dom_ids[i], SerializedDomain{}).first->second;

        const auto node_msg_size = sizes[i * 4];
        domain.element_nodes.resize(node_msg_size);
        messages.emplace_back(comm.receiveAsync(domain.element_nodes.data(), node_msg_size, source, msg_tag++));

        const auto data_msg_size = sizes[i * 4 + 1];
        domain.element_data.resize(data_msg_size);
        messages.emplace_back(comm.receiveAsync(domain.element_data.data(), data_msg_size, source, msg_tag++));

        const auto id_msg_size = sizes[i * 4 + 2];
        domain.element_ids.resize(id_msg_size);
        messages.emplace_back(comm.receiveAsync(domain.element_ids.data(), id_msg_size, source, msg_tag++));

        const auto offset_msg_size = sizes[i * 4 + 3];
        domain.type_order_offsets.resize(offset_msg_size);
        domain.types.resize(offset_msg_size);
        domain.orders.resize(offset_msg_size);
        messages.emplace_back(comm.receiveAsync(domain.type_order_offsets.data(), offset_msg_size, source, msg_tag++));
        messages.emplace_back(comm.receiveAsync(domain.types.data(), offset_msg_size, source, msg_tag++));
        messages.emplace_back(comm.receiveAsync(domain.orders.data(), offset_msg_size, source, msg_tag++));
    }

    const auto nodes_size = sizes[4 * n_doms];
    ret_val.nodes.resize(nodes_size);
    messages.emplace_back(comm.receiveAsync(ret_val.nodes.data(), nodes_size, source, msg_tag++));

    const auto ghost_nodes_size = sizes.back();
    ret_val.ghost_nodes.resize(ghost_nodes_size);
    messages.emplace_back(comm.receiveAsync(ret_val.ghost_nodes.data(), ghost_nodes_size, source, msg_tag++));

    return ret_val;
}
} // namespace lstr
#endif // L3STER_COMM_DISTRIBUTEMESH_HPP
