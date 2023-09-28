#ifndef L3STER_COMM_SENDMESH_HPP
#define L3STER_COMM_SENDMESH_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/mesh/SerializeMesh.hpp"

#include <memory>
#include <utility>

namespace lstr::comm
{
inline auto sendNDoms(const MpiComm& comm, const mesh::SerializedPartition& part, int destination, int tag)
{
    auto n_dom_ptr = std::make_unique< size_t >(part.domains.size()); // extend lifetime past fun call
    auto msg       = comm.sendAsync(std::span{n_dom_ptr.get(), 1}, destination, tag);
    return std::make_pair(std::move(n_dom_ptr), std::move(msg));
}

inline auto sendIds(const MpiComm& comm, const mesh::SerializedPartition& part, int destination, int tag)
{
    std::vector< d_id_t > dom_ids(part.domains.size());
    std::ranges::transform(part.domains, begin(dom_ids), [](const auto& pair) { return pair.first; });
    auto msg = comm.sendAsync(dom_ids, destination, tag);
    return std::make_pair(std::move(dom_ids), std::move(msg));
}

inline size_t getSizeMsgLength(size_t n_domains)
{
    constexpr size_t n_domain_size_params = 4;
    return n_domain_size_params * n_domains + 3; // 4 per domain + 2 for (ghost) nodes + 1 for boundary IDs
}

inline auto sendSizes(const MpiComm& comm, const mesh::SerializedPartition& part, int destination, int tag)
{
    const auto msg_size  = getSizeMsgLength(part.domains.size());
    auto       size_info = std::vector< size_t >{};
    size_info.reserve(msg_size);
    for (const auto& dom : part.domains)
    {
        size_info.push_back(dom.second.element_nodes.size());
        size_info.push_back(dom.second.element_data.size());
        size_info.push_back(dom.second.element_ids.size());
        size_info.push_back(dom.second.type_order_offsets.size());
    }
    size_info.push_back(part.nodes.size());
    size_info.push_back(part.n_owned_nodes);
    size_info.push_back(part.boundaries.size());
    auto msg = comm.sendAsync(size_info, destination, tag);
    return std::make_pair(std::move(size_info), std::move(msg));
}

inline void sendPartition(const MpiComm& comm, const mesh::SerializedPartition& part, int destination)
{
    int msg_tag = 0;

    const size_t n_messages = 6 * part.domains.size() + 2; // 6 per domain + nodes + boundary IDs
    auto         requests   = std::vector< MpiComm::Request >{};
    requests.reserve(n_messages);

    // prelims
    const auto n_dom_msg_and_data = sendNDoms(comm, part, destination, msg_tag++);
    const auto ids_msg_and_data   = sendIds(comm, part, destination, msg_tag++);
    const auto size_msg_and_data  = sendSizes(comm, part, destination, msg_tag++);

    // domain data
    for (const auto& [id, dom] : part.domains)
    {
        requests.emplace_back(comm.sendAsync(dom.element_nodes, destination, msg_tag++));
        requests.emplace_back(comm.sendAsync(dom.element_data, destination, msg_tag++));
        requests.emplace_back(comm.sendAsync(dom.element_ids, destination, msg_tag++));
        requests.emplace_back(comm.sendAsync(dom.type_order_offsets, destination, msg_tag++));
        requests.emplace_back(comm.sendAsync(dom.types, destination, msg_tag++));
        requests.emplace_back(comm.sendAsync(dom.orders, destination, msg_tag++));
    }

    // nodes
    requests.emplace_back(comm.sendAsync(part.nodes, destination, msg_tag++));

    // boundaries
    requests.emplace_back(comm.sendAsync(part.boundaries, destination, msg_tag++));
    MpiComm::Request::waitAll(requests);
}
} // namespace lstr::comm
#endif // L3STER_COMM_SENDMESH_HPP
