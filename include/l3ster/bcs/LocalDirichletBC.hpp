#ifndef L3STER_BCS_LOCALDIRICHLETBC_HPP
#define L3STER_BCS_LOCALDIRICHLETBC_HPP

#include "l3ster/bcs/BCDefinition.hpp"
#include "l3ster/comm/ImportExport.hpp"
#include "l3ster/common/KernelInterface.hpp"
#include "l3ster/common/ProblemDefinition.hpp"
#include "l3ster/dofs/NodeToDofMap.hpp"
#include "l3ster/mesh/LocalMeshView.hpp"

namespace lstr::bcs
{
class LocalDirichletBC
{
public:
    LocalDirichletBC() = default;
    template < size_t max_dofs_per_node, el_o_t... orders >
    LocalDirichletBC(const dofs::LocalDofMap< max_dofs_per_node >&             node_dof_map,
                     const mesh::LocalMeshView< orders... >&                   interior_mesh,
                     const mesh::LocalMeshView< orders... >&                   border_mesh,
                     const MpiComm&                                            comm,
                     const std::shared_ptr< const comm::ImportExportContext >& comm_context,
                     const DirichletBCDefinition< max_dofs_per_node >&         bc_def);

    bool isEmpty() const { return m_dofs_set.empty(); }
    bool isDirichletDof(local_dof_t dof) const { return m_dofs_set.contains(dof); }
    auto getOwnedDirichletDofs() const -> std::span< const local_dof_t > { return m_dofs_sorted; }

private:
    robin_hood::unordered_flat_set< local_dof_t > m_dofs_set;
    util::ArrayOwner< local_dof_t >               m_dofs_sorted;
};

template < size_t max_dofs_per_node, el_o_t... orders >
LocalDirichletBC::LocalDirichletBC(const dofs::LocalDofMap< max_dofs_per_node >&             node_dof_map,
                                   const mesh::LocalMeshView< orders... >&                   interior_mesh,
                                   const mesh::LocalMeshView< orders... >&                   border_mesh,
                                   const MpiComm&                                            comm,
                                   const std::shared_ptr< const comm::ImportExportContext >& comm_context,
                                   const DirichletBCDefinition< max_dofs_per_node >&         bc_def)
{
    if (bc_def.empty())
        return;

    const auto num_dofs_owned     = node_dof_map.getNumOwnedDofs();
    const auto num_dofs_total     = node_dof_map.getNumTotalDofs();
    auto       dirichlet_dofs_bmp = util::ArrayOwner< char >(num_dofs_total, false);
    const auto mark_side_dofs     = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::LocalElementView< ET, EO >& el) {
        const auto& el_nodes = el.getLocalNodes();
        for (const auto& [side, domain] : el.getBoundaries())
        {
            auto dof_bmp = std::bitset< max_dofs_per_node >{};
            for (const auto& [domains, dof_inds] : bc_def)
                if (std::ranges::binary_search(domains, domain))
                    for (auto i : dof_inds)
                        dof_bmp.set(i);
            const auto dof_inds = util::getTrueInds(dof_bmp);
            for (auto node_ind : mesh::getSideNodeIndices< ET, EO >(side))
            {
                const auto  node_lid  = el_nodes[node_ind];
                const auto& node_dofs = node_dof_map(node_lid);
                for (auto dof : dof_inds | std::views::transform([&](auto i) { return node_dofs[i]; }) |
                                    std::views::filter(dofs::LocalDofMap< max_dofs_per_node >::isValid))
                    std::atomic_ref{dirichlet_dofs_bmp.at(dof)}.store(true, std::memory_order_relaxed);
            }
        }
    };
    const auto mark_dirichlet_dofs = [&](const mesh::LocalMeshView< orders... >& mesh) {
        for (const auto& [_, domain] : mesh.getDomains())
            domain.elements.visit(mark_side_dofs, std::execution::par);
    };

    auto importer = comm::Import< char >{comm_context, 1};
    auto exporter = comm::Export< char >{comm_context, 1};
    importer.setOwned(dirichlet_dofs_bmp, num_dofs_owned);
    exporter.setOwned(dirichlet_dofs_bmp, num_dofs_owned);
    importer.setShared(std::span{dirichlet_dofs_bmp}.subspan(num_dofs_owned), num_dofs_total);
    exporter.setShared(std::span{dirichlet_dofs_bmp}.subspan(num_dofs_owned), num_dofs_total);

    exporter.postRecvs(comm);
    mark_dirichlet_dofs(border_mesh);
    exporter.postSends(comm);
    mark_dirichlet_dofs(interior_mesh);
    exporter.wait([](char& flag, char update) { std::atomic_ref{flag}.fetch_or(update, std::memory_order_relaxed); });
    importer.postComms(comm);
    importer.wait();

    m_dofs_set = robin_hood::unordered_flat_set< local_dof_t >(std::ranges::count(dirichlet_dofs_bmp, true));
    for (const auto& [i, flag] : dirichlet_dofs_bmp | std::views::enumerate)
        if (flag)
            m_dofs_set.insert(static_cast< local_dof_t >(i));
    const auto num_dofs_owned_signed = static_cast< local_dof_t >(num_dofs_owned);
    auto owned_dofs = m_dofs_set | std::views::filter([&](local_dof_t dof) { return dof < num_dofs_owned_signed; });
    m_dofs_sorted   = util::ArrayOwner< local_dof_t >(owned_dofs);
    std::ranges::sort(m_dofs_sorted);
}
} // namespace lstr::bcs
#endif // L3STER_BCS_LOCALDIRICHLETBC_HPP
