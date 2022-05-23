#ifndef L3STER_BCS_GETDIRICHLETDOFS_HPP
#define L3STER_BCS_GETDIRICHLETDOFS_HPP

#include "l3ster/global_assembly/NodeToDofMap.hpp"
#include "l3ster/mesh/BoundaryView.hpp"

namespace lstr
{
template < detail::ProblemDef_c auto dirichlet_def >
auto getDirichletDofs(const MeshPartition&                                        mesh,
                      const NodeToDofMap< detail::deduceNFields(dirichlet_def) >& dof_map,
                      ConstexprValue< dirichlet_def >                             dirichlet_def_ctwrapper)
{
    std::vector< global_dof_t > owned_dbcs, shared_dbcs;
    const auto                  process_domain = [&]< auto domain_def >(ConstexprValue< domain_def >)
    {
        constexpr auto  domain_id        = domain_def.first;
        constexpr auto& coverage         = domain_def.second;
        constexpr auto  covered_dof_inds = getTrueInds< coverage >();
        const auto      get_node_dofs    = [&](n_id_t node) {
            std::array< global_dof_t, std::tuple_size_v< decltype(covered_dof_inds) > > dofs;
            const auto&                                                                 full_node_dofs = dof_map(node);
            for (auto insert_it = dofs.begin(); auto ind : covered_dof_inds)
                *insert_it++ = full_node_dofs[ind];
            return dofs;
        };
        const auto process_node = [&](n_id_t node) {
            const auto node_dofs = get_node_dofs(node);
            if (std::ranges::binary_search(mesh.getGhostNodes(), node))
                std::ranges::copy(node_dofs, std::back_inserter(shared_dbcs));
            else
                std::ranges::copy(node_dofs, std::back_inserter(owned_dbcs));
        };
        const auto process_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
            for (auto node : element.getNodes())
                process_node(node);
        };
        mesh.cvisit(process_element, {domain_id});
    };
    forConstexpr(process_domain, dirichlet_def_ctwrapper);

    constexpr auto uniquify_vec = [](std::vector< global_dof_t >& vec) {
        std::ranges::sort(vec);
        const auto erase_range = std::ranges::unique(vec);
        vec.erase(begin(erase_range), end(erase_range));
        vec.shrink_to_fit();
    };
    uniquify_vec(owned_dbcs);
    uniquify_vec(shared_dbcs);
    return std::make_pair(std::move(owned_dbcs), std::move(shared_dbcs));
}
} // namespace lstr
#endif // L3STER_BCS_GETDIRICHLETDOFS_HPP
