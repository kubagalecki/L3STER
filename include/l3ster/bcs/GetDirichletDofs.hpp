#ifndef L3STER_BCS_GETDIRICHLETDOFS_HPP
#define L3STER_BCS_GETDIRICHLETDOFS_HPP

#include "l3ster/global_assembly/SparsityGraph.hpp"

namespace lstr::detail
{
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
auto getDirichletDofs(const MeshPartition&                                      mesh,
                      const Teuchos::RCP< const Tpetra::FECrsGraph<> >&         sparsity_graph,
                      const NodeToDofMap< detail::deduceNFields(problem_def) >& dof_map,
                      ConstexprValue< problem_def >                             problem_def_ctwrapper,
                      ConstexprValue< dirichlet_def >                           dirichlet_def_ctwrapper)
{
    const auto mark_owned_dirichlet_dofs = [&] {
        auto dirichlet_dofs = makeTeuchosRCP< Tpetra::FEMultiVector<> >(
            sparsity_graph->getColMap(), sparsity_graph->getImporter(), size_t{1});
        dirichlet_dofs->beginAssembly();
        const auto process_domain = [&]< auto domain_def >(ConstexprValue< domain_def >)
        {
            constexpr auto  domain_id        = domain_def.first;
            constexpr auto& coverage         = domain_def.second;
            constexpr auto  covered_dof_inds = getTrueInds< coverage >();
            const auto      process_element  = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
                const auto el_dirichlet_dofs = getUnsortedElementDofs< covered_dof_inds >(element, dof_map);
                for (auto dof : el_dirichlet_dofs)
                    dirichlet_dofs->replaceGlobalValue(dof, 0, 1.);
            };
            mesh.cvisit(process_element, {domain_id});
        };
        forConstexpr(process_domain, dirichlet_def_ctwrapper);
        dirichlet_dofs->endAssembly();
        return dirichlet_dofs;
    };
    const auto mark_dirichlet_dof_cols = [&](const Teuchos::RCP< const Tpetra::Vector<> >& owned_dirichlet_dofs) {
        const auto dirichlet_dof_cols = makeTeuchosRCP< Tpetra::Vector<> >(sparsity_graph->getColMap());
        const auto importer           = Tpetra::Import<>{owned_dirichlet_dofs->getMap(), dirichlet_dof_cols->getMap()};
        dirichlet_dof_cols->doImport(*owned_dirichlet_dofs, importer, Tpetra::REPLACE);
        return dirichlet_dof_cols;
    };
    constexpr auto extract_marked_dofs = [](const Teuchos::RCP< const Tpetra::Vector<> >& marked_dofs) {
        constexpr auto is_marked = [](val_t v) {
            return v > .5;
        };
        const auto                  entries = marked_dofs->getData();
        const auto                  n_ones  = std::ranges::count_if(entries, is_marked);
        std::vector< global_dof_t > retval;
        retval.reserve(n_ones);
        for (local_dof_t local_dof = 0; auto v : entries)
        {
            if (is_marked(v))
                retval.push_back(marked_dofs->getMap()->getGlobalElement(local_dof));
            ++local_dof;
        }
        std::ranges::sort(retval);
        return retval;
    };

    const auto marked_owned_dirichlet_dofs = mark_owned_dirichlet_dofs();
    const auto marked_col_dirichlet_dofs   = mark_dirichlet_dof_cols(marked_owned_dirichlet_dofs->getVector(0));
    return std::make_pair(extract_marked_dofs(marked_owned_dirichlet_dofs->getVector(0)),
                          extract_marked_dofs(marked_col_dirichlet_dofs));
}
} // namespace lstr::detail
#endif // L3STER_BCS_GETDIRICHLETDOFS_HPP
