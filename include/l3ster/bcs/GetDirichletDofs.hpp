#ifndef L3STER_BCS_GETDIRICHLETDOFS_HPP
#define L3STER_BCS_GETDIRICHLETDOFS_HPP

#include "l3ster/bcs/BCDefinition.hpp"
#include "l3ster/dofs/DofsFromNodes.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

namespace lstr::bcs
{
struct DirichletDofs
{
    std::vector< global_dof_t > owned, shared;
};

template < el_o_t... orders, size_t max_dofs_per_node >
auto getDirichletDofs(const mesh::MeshPartition< orders... >&              mesh,
                      const Teuchos::RCP< const tpetra_fecrsgraph_t >&     sparsity_graph,
                      const dofs::NodeToGlobalDofMap< max_dofs_per_node >& node_to_dof_map,
                      const DirichletBCDefinition< max_dofs_per_node >&    bc_def) -> DirichletDofs
{
    const auto mark_owned_dirichlet_dofs = [&] {
        const auto dirichlet_dofs = util::makeTeuchosRCP< tpetra_femultivector_t >(
            sparsity_graph->getColMap(), sparsity_graph->getImporter(), 1u);
        dirichlet_dofs->beginAssembly();
        for (const auto& [domains, dof_inds] : bc_def)
        {
            const auto process_element = [&]< mesh::ElementType T, el_o_t O >(const mesh::Element< T, O >& element) {
                for (auto dof : dofs::getDofsFromNodes(element.getNodes(), node_to_dof_map, dof_inds))
                    dirichlet_dofs->replaceGlobalValue(dof, 0, 1.);
            };
            mesh.visit(process_element, domains);
        }
        dirichlet_dofs->endAssembly();
        return dirichlet_dofs;
    };
    const auto mark_dirichlet_dof_cols = [&](const Teuchos::RCP< const tpetra_vector_t >& owned_dirichlet_dofs) {
        const auto dirichlet_dof_cols = util::makeTeuchosRCP< tpetra_vector_t >(sparsity_graph->getColMap());
        const auto importer           = tpetra_import_t{owned_dirichlet_dofs->getMap(), dirichlet_dof_cols->getMap()};
        dirichlet_dof_cols->doImport(*owned_dirichlet_dofs, importer, Tpetra::REPLACE);
        return dirichlet_dof_cols;
    };
    constexpr auto extract_marked_dofs = [](const Teuchos::RCP< const tpetra_vector_t >& marked_dofs) {
        constexpr auto is_marked = [](val_t v) {
            return v > .5;
        };
        const auto& marked_dofs_map = *marked_dofs->getMap();
        const auto  entries         = marked_dofs->getLocalViewHost(Tpetra::Access::ReadOnly);
        const auto  entries_span    = util::asSpan(Kokkos::subview(entries, Kokkos::ALL, 0));
        const auto  n_ones          = std::ranges::count_if(entries_span, is_marked);
        auto        retval          = std::vector< global_dof_t >{};
        retval.reserve(n_ones);
        for (local_dof_t local_dof = 0; auto v : entries_span)
        {
            if (is_marked(v))
                retval.push_back(marked_dofs_map.getGlobalElement(local_dof));
            ++local_dof;
        }
        std::ranges::sort(retval);
        retval.shrink_to_fit();
        return retval;
    };

    const auto marked_owned_dirichlet_dofs = mark_owned_dirichlet_dofs();
    const auto marked_col_dirichlet_dofs   = mark_dirichlet_dof_cols(marked_owned_dirichlet_dofs->getVector(0));
    return {extract_marked_dofs(marked_owned_dirichlet_dofs->getVector(0)),
            extract_marked_dofs(marked_col_dirichlet_dofs)};
}
} // namespace lstr::bcs
#endif // L3STER_BCS_GETDIRICHLETDOFS_HPP
