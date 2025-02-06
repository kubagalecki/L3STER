#ifndef L3STER_ALGSYS_SPARSITYGRAPH_HPP
#define L3STER_ALGSYS_SPARSITYGRAPH_HPP

#include "l3ster/dofs/DofsFromNodes.hpp"
#include "l3ster/dofs/NeighborManager.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/CrsGraph.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/IndexMap.hpp"
#include "l3ster/util/StaticVector.hpp"

namespace lstr::algsys
{
namespace detail
{
using row_to_entries_map_t = robin_hood::unordered_flat_map< global_dof_t, std::vector< global_dof_t > >;

struct LocalGraphGID
{
    util::CrsGraph< global_dof_t >    graph;
    util::ArrayOwner< std::uint32_t > row_sizes;

    auto operator()(size_t row) const { return graph(row).first(row_sizes.at(row)); }
};
template < el_o_t... orders, size_t max_dofs_per_node, CondensationPolicy CP, size_t n_domains >
auto computeLocalGraph(const mesh::MeshPartition< orders... >&              mesh,
                       const dofs::NodeToGlobalDofMap< max_dofs_per_node >& node_to_dof_map,
                       const dofs::NodeCondensationMap< CP >&               cond_map,
                       const ProblemDef< n_domains, max_dofs_per_node >&    problem_def) -> LocalGraphGID
{
    L3STER_PROFILE_FUNCTION;
    const auto& ownership                 = node_to_dof_map.ownership();
    const auto  num_dofs_local            = ownership.localSize();
    auto        row_sizes                 = util::ArrayOwner< std::uint32_t >(num_dofs_local, 0);
    const auto  count_row_sizes_overalloc = [&](const auto& domain_def) {
        const auto& [domain, dof_bmp] = domain_def;
        const auto dof_inds           = util::getTrueInds(dof_bmp);
        const auto inds_span          = std::span{dof_inds};
        const auto visit_el           = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& el) {
            const auto el_dofs     = dofs::getDofsCopy(cond_map, node_to_dof_map, el, inds_span);
            const auto num_el_dofs = static_cast< std::uint32_t >(el_dofs.size());
            for (auto dof : el_dofs)
            {
                const auto lid = ownership.getLocalIndex(dof);
                std::atomic_ref{row_sizes.at(lid)}.fetch_add(num_el_dofs, std::memory_order_relaxed);
            }
        };
        mesh.visit(visit_el, domain, std::execution::par);
    };
    util::tbb::parallelFor(problem_def, count_row_sizes_overalloc);
    auto       local_graph               = util::CrsGraph< global_dof_t >(row_sizes);
    const auto fill_rows_with_duplicates = [&](const auto& domain_def) {
        const auto& [domain, dof_bmp] = domain_def;
        const auto dof_inds           = util::getTrueInds(dof_bmp);
        const auto inds_span          = std::span{dof_inds};
        const auto visit_el           = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& el) {
            const auto el_dofs     = dofs::getDofsCopy(cond_map, node_to_dof_map, el, inds_span);
            const auto num_el_dofs = static_cast< std::uint32_t >(el_dofs.size());
            for (auto dof : el_dofs)
            {
                const auto lid  = ownership.getLocalIndex(dof);
                const auto prev = std::atomic_ref{row_sizes.at(lid)}.fetch_sub(num_el_dofs, std::memory_order_acq_rel);
                const auto graph_row  = local_graph(lid);
                const auto copy_begin = graph_row.size() - prev;
                const auto dest       = local_graph(lid).subspan(copy_begin, num_el_dofs);
                std::ranges::copy(el_dofs, dest.begin());
            }
        };
        mesh.visit(visit_el, domain, std::execution::par);
    };
    util::tbb::parallelFor(problem_def, fill_rows_with_duplicates);
    util::throwingAssert(std::ranges::none_of(row_sizes, std::identity{}));
    const auto sort_row_remove_dup = [&](std::uint32_t lid) {
        const auto row = local_graph(lid);
        std::ranges::sort(row);
        const auto num_repeated = std::ranges::size(std::ranges::unique(row));
        row_sizes[lid]          = static_cast< std::uint32_t >(row.size() - num_repeated);
    };
    util::tbb::parallelFor(std::views::iota(0u, num_dofs_local), sort_row_remove_dup);
    return {std::move(local_graph), std::move(row_sizes)};
}

inline auto serializeSharedGraph(std::span< const global_dof_t >                 rows_to_serialize,
                                 const util::SegmentedOwnership< global_dof_t >& ownership,
                                 const LocalGraphGID& local_graph) -> util::ArrayOwner< global_dof_t >
{
    const auto num_entries =
        std::transform_reduce(rows_to_serialize.begin(), rows_to_serialize.end(), 0uz, std::plus{}, [&](auto row) {
            const auto lid = ownership.getLocalIndex(row);
            return local_graph(lid).size();
        });
    const auto serial_size = num_entries + 2 * rows_to_serialize.size() + 1;
    auto       retval      = util::ArrayOwner< global_dof_t >(serial_size);
    auto       write_iter  = retval.begin();
    *write_iter++          = std::ranges::ssize(rows_to_serialize);
    write_iter             = std::ranges::transform(rows_to_serialize, write_iter, [&](auto row) {
                     const auto lid = ownership.getLocalIndex(row);
                     return std::ranges::ssize(local_graph(lid));
                 }).out;
    write_iter             = std::ranges::copy(rows_to_serialize, write_iter).out;
    for (auto row : rows_to_serialize | std::views::transform([&](auto row) { return ownership.getLocalIndex(row); }))
        write_iter = std::ranges::copy(local_graph(row), write_iter).out;
    util::throwingAssert(write_iter == retval.end());
    return retval;
}

inline void appendSharedFromSerial(std::span< const global_dof_t >                 serial,
                                   const LocalGraphGID&                            local_graph,
                                   const util::SegmentedOwnership< global_dof_t >& ownership,
                                   row_to_entries_map_t&                           current)
{
    util::throwingAssert(not serial.empty());
    const auto   num_rows    = static_cast< size_t >(serial.front());
    const size_t header_size = 2 * num_rows + 1;
    util::throwingAssert(serial.size() >= header_size);
    const auto row_sizes   = serial.subspan(1, num_rows);
    const auto rows        = serial.subspan(1 + num_rows, num_rows);
    auto       row_offsets = util::ArrayOwner< size_t >(num_rows);
    std::exclusive_scan(row_sizes.begin(), row_sizes.end(), row_offsets.begin(), header_size);
    util::throwingAssert(std::ranges::all_of(std::views::zip_transform(std::plus{}, row_offsets, row_sizes),
                                             [&](auto bound) { return bound <= serial.size(); }));
    util::throwingAssert(std::unordered_set(rows.begin(), rows.end()).size() == rows.size());
    for (const auto& [row, row_size] : std::views::zip(rows, row_sizes))
    {
        auto& row_entries = current[row];
        row_entries.reserve(row_entries.size() + static_cast< size_t >(row_size));
    }
    const auto zipped_view = std::views::zip(rows, row_offsets, row_sizes);
    const auto insert_row  = [&](const auto& zipped) {
        const auto& [row, offset, size] = zipped;
        const auto row_lid              = ownership.getLocalIndex(row);
        const auto local_entries        = local_graph(row_lid);
        const auto new_entries          = serial.subspan(offset, static_cast< size_t >(size));
        auto&      row_entries          = current.at(row);
        const auto old_size             = row_entries.size();
        std::ranges::set_difference(new_entries, local_entries, std::back_inserter(row_entries));
        if (old_size != 0)
            util::sortRemoveDup(row_entries);
    };
    util::tbb::parallelFor(zipped_view, insert_row);
}

// Who am I sending my shared rows to?
template < typename DofToOwnerMap >
auto computeOwnerToSharedMap(const util::SegmentedOwnership< global_dof_t >& ownership,
                             const DofToOwnerMap&                            dof_to_owner)
    -> robin_hood::unordered_flat_map< int, std::vector< global_dof_t > >
{
    auto retval = robin_hood::unordered_flat_map< int, std::vector< global_dof_t > >{};
    for (auto shared : ownership.shared())
        retval[dof_to_owner(shared)].push_back(shared);
    return retval;
}

// Which ranks are sending me data from their shared rows?
inline auto computeInNbrs(const MpiComm&                                                            comm,
                          const robin_hood::unordered_flat_map< int, std::vector< global_dof_t > >& owner_to_shared)
    -> std::vector< int >
{
    const auto               comm_sz = static_cast< size_t >(comm.getSize());
    util::ArrayOwner< char > owns_my_shared_bmp(comm_sz, false), am_shared_owner_bmp(comm_sz);
    for (int dest : owner_to_shared | std::views::transform([](const auto& p) { return p.first; }))
        owns_my_shared_bmp.at(static_cast< size_t >(dest)) = true;
    comm.allToAllAsync(owns_my_shared_bmp, am_shared_owner_bmp).wait();
    auto retval = std::vector< int >{};
    std::ranges::copy_if(std::views::iota(0, comm.getSize()), std::back_inserter(retval), [&](int rank) {
        return am_shared_owner_bmp[static_cast< size_t >(rank)];
    });
    return retval;
}

// Which entries are my in-neighbors inserting into my graph?
inline auto computeInNbrData(const MpiComm&                                                            comm,
                             const util::SegmentedOwnership< global_dof_t >&                           ownership,
                             const LocalGraphGID&                                                      local_graph,
                             const std::vector< int >&                                                 in_nbrs,
                             const robin_hood::unordered_flat_map< int, std::vector< global_dof_t > >& owner_to_shared)
    -> row_to_entries_map_t
{
    L3STER_PROFILE_FUNCTION;
    // Send
    auto send_reqs = std::vector< MpiComm::Request >{};
    auto send_data = std::vector< util::ArrayOwner< global_dof_t > >{};
    send_reqs.reserve(owner_to_shared.size());
    send_data.reserve(owner_to_shared.size());
    for (const auto& [dest_rank, rows] : owner_to_shared)
    {
        auto& data = send_data.emplace_back(serializeSharedGraph(rows, ownership, local_graph));
        send_reqs.push_back(comm.sendAsync(data, dest_rank, 0));
    }

    // Receive
    auto retval   = row_to_entries_map_t{};
    auto recv_buf = std::vector< global_dof_t >{};
    auto received = util::DynamicBitset{in_nbrs.size()};
    while (received.count() != in_nbrs.size())
        for (auto&& [i, nbr] : in_nbrs | std::views::enumerate)
            if (not received.test(static_cast< size_t >(i)))
                if (const auto& [status, ready] = comm.probeAsync(nbr, 0); ready)
                {
                    recv_buf.resize(static_cast< size_t >(status.numElems< global_dof_t >()));
                    comm.receive(recv_buf, nbr, 0);
                    appendSharedFromSerial(recv_buf, local_graph, ownership, retval);
                    received.set(static_cast< size_t >(i));
                }

    MpiComm::Request::waitAll(send_reqs);
#ifdef L3STER_PROFILE_EXECUTION
    comm.barrier();
#endif
    return retval;
}

inline auto makeDofToOwnerMap(std::span< const global_dof_t > ownership_dist)
{
    return [ownership_dist](global_dof_t dof) {
        return static_cast< int >(std::distance(ownership_dist.begin(), std::ranges::upper_bound(ownership_dist, dof)));
    };
}

inline auto makeColumnOwnership(const util::SegmentedOwnership< global_dof_t >& row_ownership,
                                const row_to_entries_map_t& in_nbr_data) -> util::SegmentedOwnership< global_dof_t >
{
    const auto owned       = row_ownership.owned();
    const auto shared_rows = row_ownership.shared();
    const auto num_owned   = owned.size();
    const auto owned_begin = num_owned != 0 ? owned.front() : 0;
    auto       shared      = std::set(shared_rows.begin(), shared_rows.end());
    for (const auto& [_, entries] : in_nbr_data)
        for (auto e : entries | std::views::filter([&](auto e) { return not row_ownership.isOwned(e); }))
            shared.insert(e);
    return {owned_begin, num_owned, shared};
}

inline auto makeTpetraMapOwned(const util::SegmentedOwnership< global_dof_t >& ownership,
                               Teuchos::RCP< const Teuchos::MpiComm< int > >   comm,
                               Tpetra::global_size_t num_dofs_global) -> Teuchos::RCP< const tpetra_map_t >
{
    return util::makeTeuchosRCP< const tpetra_map_t >(num_dofs_global, ownership.owned().size(), 0, std::move(comm));
}

inline auto makeTpetraMapOwnedPlusShared(const util::SegmentedOwnership< global_dof_t >& ownership,
                                         Teuchos::RCP< const Teuchos::MpiComm< int > >   comm)
    -> Teuchos::RCP< const tpetra_map_t >
{
    auto       dofs      = util::ArrayOwner< global_dof_t >(ownership.localSize());
    const auto owned_end = std::ranges::copy(ownership.owned(), dofs.begin()).out;
    std::ranges::copy(ownership.shared(), owned_end);
    const auto dofs_teuchos_view = util::asTeuchosView(dofs);
    const auto unknown_num_dofs  = Teuchos::OrdinalTraits< Tpetra::global_size_t >::invalid(); // Tpetra does allreduce
    return util::makeTeuchosRCP< const tpetra_map_t >(unknown_num_dofs, dofs_teuchos_view, 0, std::move(comm));
}

void insertIntoTpetraGraph(const LocalGraphGID&                            local_graph,
                           const row_to_entries_map_t&                     shared_entries,
                           const util::SegmentedOwnership< global_dof_t >& row_ownership,
                           const util::SegmentedOwnership< global_dof_t >& col_ownership,
                           tpetra_crsgraph_t&                              tpetra_graph)
{
    L3STER_PROFILE_FUNCTION;
    // Construct hashmap instead of calling col_ownership.getLocalIndex()
    const auto col_g2l = [&,
                          num_owned  = col_ownership.owned().size(),
                          shared_inv = util::IndexMap{col_ownership.shared()}](global_dof_t col) {
        return col_ownership.isOwned(col) ? col_ownership.getLocalIndex(col) : shared_inv(col) + num_owned;
    };
    auto lids_to_insert = std::vector< local_dof_t >{};
    for (size_t row_lid = 0; row_lid != local_graph.row_sizes.size(); ++row_lid)
    {
        const auto row_gid = row_ownership.getGlobalIndex(row_lid);
        lids_to_insert.clear();
        std::ranges::transform(local_graph(row_lid), std::back_inserter(lids_to_insert), std::cref(col_g2l));
        if (const auto shared_iter = shared_entries.find(row_gid); shared_iter != shared_entries.end())
            std::ranges::transform(shared_iter->second, std::back_inserter(lids_to_insert), std::cref(col_g2l));
        std::ranges::sort(lids_to_insert);
        tpetra_graph.insertLocalIndices(static_cast< local_dof_t >(row_lid), util::asTeuchosView(lids_to_insert));
    }
}

auto makeRowSizes(const LocalGraphGID&                            local_graph,
                  const row_to_entries_map_t&                     shared_entries,
                  const util::SegmentedOwnership< global_dof_t >& row_ownership)
{
    using return_t       = Kokkos::DualView< size_t*, tpetra_fecrsgraph_t::execution_space >;
    const auto num_rows  = local_graph.row_sizes.size();
    auto       retval    = return_t("Rows sizes", num_rows);
    const auto host_view = retval.view_host();
    retval.modify_host();
    for (size_t row_lid = 0; row_lid != num_rows; ++row_lid)
        host_view(row_lid) = local_graph(row_lid).size();
    for (const auto& [row_gid, entries] : shared_entries)
        host_view(row_ownership.getLocalIndex(row_gid)) += entries.size();
    retval.sync_device();
    return retval;
}
} // namespace detail

template < el_o_t... orders, ProblemDef problem_def, CondensationPolicy CP >
auto makeSparsityGraph(const MpiComm&                                          comm,
                       const mesh::MeshPartition< orders... >&                 mesh,
                       const dofs::NodeToGlobalDofMap< problem_def.n_fields >& node_to_dof_map,
                       const dofs::NodeCondensationMap< CP >&                  cond_map,
                       util::ConstexprValue< problem_def >) -> Teuchos::RCP< const tpetra_fecrsgraph_t >
{
    L3STER_PROFILE_FUNCTION;
    const auto local_graph     = detail::computeLocalGraph(mesh, node_to_dof_map, cond_map, problem_def);
    const auto ownership_dist  = node_to_dof_map.ownership().getOwnershipDist(comm);
    const auto num_dofs_global = static_cast< size_t >(ownership_dist.back());
    const auto dof2owner       = detail::makeDofToOwnerMap(ownership_dist);
    const auto owner2shared    = detail::computeOwnerToSharedMap(node_to_dof_map.ownership(), dof2owner);
    const auto in_nbrs         = detail::computeInNbrs(comm, owner2shared);
    const auto in_nbr_data =
        detail::computeInNbrData(comm, node_to_dof_map.ownership(), local_graph, in_nbrs, owner2shared);
    const auto col_ownership = detail::makeColumnOwnership(node_to_dof_map.ownership(), in_nbr_data);
    const auto teuchos_comm  = util::makeTeuchosRCP< const Teuchos::MpiComm< int > >(comm.get());
    const auto row_map_owned = detail::makeTpetraMapOwned(node_to_dof_map.ownership(), teuchos_comm, num_dofs_global);
    const auto row_map_all   = detail::makeTpetraMapOwnedPlusShared(node_to_dof_map.ownership(), teuchos_comm);
    const auto col_map       = detail::makeTpetraMapOwnedPlusShared(col_ownership, teuchos_comm);
    const auto row_sizes     = detail::makeRowSizes(local_graph, in_nbr_data, node_to_dof_map.ownership());
    auto       retval = util::makeTeuchosRCP< tpetra_fecrsgraph_t >(row_map_owned, row_map_all, col_map, row_sizes);
    retval->beginAssembly();
    detail::insertIntoTpetraGraph(local_graph, in_nbr_data, node_to_dof_map.ownership(), col_ownership, *retval);
    retval->endAssembly();
    return retval;
}
} // namespace lstr::algsys
#endif // L3STER_ALGSYS_SPARSITYGRAPH_HPP
