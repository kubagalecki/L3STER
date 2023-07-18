#ifndef L3STER_BCS_GETDIRICHLETDOFS_HPP
#define L3STER_BCS_GETDIRICHLETDOFS_HPP

#include "l3ster/dofs/DofsFromNodes.hpp"

namespace lstr::bcs
{
template < el_o_t... orders, CondensationPolicy CP, ProblemDef_c auto problem_def, ProblemDef_c auto dirichlet_def >
auto getDirichletDofs(const mesh::MeshPartition< orders... >&                               mesh,
                      const Teuchos::RCP< const tpetra_fecrsgraph_t >&                      sparsity_graph,
                      const dofs::NodeToGlobalDofMap< detail::deduceNFields(problem_def) >& node_to_dof_map,
                      const dofs::NodeCondensationMap< CP >&                                cond_map,
                      util::ConstexprValue< problem_def >,
                      util::ConstexprValue< dirichlet_def > dirichletdef_ctwrpr)
{
    const auto mark_owned_dirichlet_dofs = [&] {
        const auto dirichlet_dofs = util::makeTeuchosRCP< tpetra_femultivector_t >(
            sparsity_graph->getColMap(), sparsity_graph->getImporter(), 1u);
        dirichlet_dofs->beginAssembly();
        const auto process_domain = [&]< auto domain_def >(util::ConstexprValue< domain_def >) {
            constexpr auto  domain_id        = domain_def.first;
            constexpr auto& coverage         = domain_def.second;
            constexpr auto  covered_dof_inds = util::getTrueInds< coverage >();
            const auto process_element = [&]< mesh::ElementType T, el_o_t O >(const mesh::Element< T, O >& element) {
                const auto element_dirichlet_dofs =
                    dofs::getUnsortedPrimaryDofs< covered_dof_inds >(element, node_to_dof_map, cond_map);
                for (auto dof : element_dirichlet_dofs)
                    dirichlet_dofs->replaceGlobalValue(dof, 0, 1.);
            };
            mesh.visit(process_element, domain_id);
        };
        util::forConstexpr(process_domain, dirichletdef_ctwrpr);
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
    return std::make_pair(extract_marked_dofs(marked_owned_dirichlet_dofs->getVector(0)),
                          extract_marked_dofs(marked_col_dirichlet_dofs));
}
} // namespace lstr::bcs
#endif // L3STER_BCS_GETDIRICHLETDOFS_HPP
