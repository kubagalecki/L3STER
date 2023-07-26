#ifndef L3STER_DOFS_DOFSFROMNODES_HPP
#define L3STER_DOFS_DOFSFROMNODES_HPP

#include "l3ster/dofs/NodeToDofMap.hpp"

namespace lstr::dofs
{
template < IndexRange_c auto dof_inds, size_t n_nodes, size_t dofs_per_node, CondensationPolicy CP >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&             nodes,
                      const dofs::NodeToGlobalDofMap< dofs_per_node >& node_dof_map,
                      const dofs::NodeCondensationMap< CP >&           cond_map,
                      util::ConstexprValue< dof_inds >                 dofinds_ctwrpr = {})
{
    std::array< global_dof_t, std::ranges::size(dof_inds) * n_nodes > retval;
    std::ranges::copy(nodes | std::views::transform([&](n_id_t node) {
                          return util::getValuesAtInds(node_dof_map(cond_map.getCondensedId(node)), dofinds_ctwrpr);
                      }) | std::views::join,
                      begin(retval));
    return retval;
}

template < size_t n_nodes, size_t dofs_per_node >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&                                    nodes,
                      const dofs::NodeToGlobalDofMap< dofs_per_node >&                        node_dof_map,
                      const dofs::NodeCondensationMap< CondensationPolicy::ElementBoundary >& cond_map)
{
    util::StaticVector< global_dof_t, dofs_per_node * n_nodes > retval;
    std::ranges::copy(
        nodes | std::views::transform([&](n_id_t node) { return node_dof_map(cond_map.getCondensedId(node)); }) |
            std::views::join | std::views::filter([](auto node) {
                return node != dofs::NodeToGlobalDofMap< dofs_per_node >::invalid_dof;
            }),
        std::back_inserter(retval));
    return retval;
}

template < IndexRange_c auto dof_inds, size_t n_nodes, size_t dofs_per_node, size_t num_maps, CondensationPolicy CP >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&                      nodes,
                      const dofs::NodeToLocalDofMap< dofs_per_node, num_maps >& node_dof_map,
                      CondensationPolicyTag< CP >,
                      const util::ConstexprValue< dof_inds > dofinds_ctwrpr = {})
{
    using dof_array_t = std::array< local_dof_t, std::ranges::size(dof_inds) * n_nodes >;
    auto retval       = std::array< dof_array_t, num_maps >{};
    auto iters        = std::array< typename dof_array_t::iterator, num_maps >{};
    std::ranges::transform(retval, begin(iters), [](auto& arr) { return arr.begin(); });
    for (auto node : nodes)
    {
        const auto& all_dof_arrays = node_dof_map(node);
        for (size_t i = 0; const auto& all_dofs : all_dof_arrays)
        {
            iters[i] = util::copyValuesAtInds(all_dofs, iters[i], dofinds_ctwrpr);
            ++i;
        }
    }
    return retval;
}

template < size_t n_nodes, size_t dofs_per_node, size_t num_maps >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&                      nodes,
                      const dofs::NodeToLocalDofMap< dofs_per_node, num_maps >& node_dof_map)
{
    using dof_vec_t = util::StaticVector< local_dof_t, dofs_per_node * n_nodes >;
    std::array< dof_vec_t, num_maps > retval;
    for (auto node : nodes)
    {
        const auto& all_dof_arrays = node_dof_map(node);
        for (size_t i = 0; const auto& all_dofs : all_dof_arrays)
        {
            std::ranges::copy_if(all_dofs, std::back_inserter(retval[i]), [](local_dof_t dof) {
                return dof != dofs::NodeToLocalDofMap< dofs_per_node, num_maps >::invalid_dof;
            });
            ++i;
        }
    }
    return retval;
}

template < IndexRange_c auto dof_inds, mesh::ElementType ET, el_o_t EO >
auto getSortedPrimaryDofs(const mesh::Element< ET, EO >&                               element,
                          const dofs::NodeToDofMap_c auto&                             node_dof_map,
                          const dofs::NodeCondensationMap< CondensationPolicy::None >& cond_map,
                          util::ConstexprValue< dof_inds >                             dofinds_ctwrpr = {})
{
    auto primary_nodes = getPrimaryNodesArray< CondensationPolicy::None >(element);
    std::ranges::sort(primary_nodes);
    return getDofsFromNodes(primary_nodes, node_dof_map, cond_map, dofinds_ctwrpr);
}

template < IndexRange_c auto  dof_inds,
           mesh::ElementType  ET,
           el_o_t             EO,
           size_t             max_dofs_per_node,
           CondensationPolicy CP >
auto getUnsortedPrimaryDofs(const mesh::Element< ET, EO >&                       element,
                            const dofs::NodeToGlobalDofMap< max_dofs_per_node >& node_dof_map,
                            const dofs::NodeCondensationMap< CP >&               cond_map,
                            util::ConstexprValue< dof_inds >                     dofinds_ctwrpr = {})
{
    return getDofsFromNodes(getPrimaryNodesArray< CP >(element), node_dof_map, cond_map, dofinds_ctwrpr);
}

template < IndexRange_c auto  dof_inds,
           mesh::ElementType  ET,
           el_o_t             EO,
           size_t             max_dofs_per_node,
           size_t             num_maps,
           CondensationPolicy CP >
auto getUnsortedPrimaryDofs(const mesh::Element< ET, EO >&                                element,
                            const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& node_dof_map,
                            CondensationPolicyTag< CP >                                   cond_policy,
                            util::ConstexprValue< dof_inds >                              dofinds_ctwrpr = {})
{
    return getDofsFromNodes(getPrimaryNodesArray< CP >(element), node_dof_map, cond_policy, dofinds_ctwrpr);
}

template < mesh::ElementType ET, el_o_t EO, size_t max_dofs_per_node, CondensationPolicy CP >
auto getUnsortedPrimaryDofs(const mesh::Element< ET, EO >&                       element,
                            const dofs::NodeToGlobalDofMap< max_dofs_per_node >& node_dof_map,
                            const dofs::NodeCondensationMap< CP >&               cond_map)
{
    return getDofsFromNodes(getPrimaryNodesArray< CP >(element), node_dof_map, cond_map);
}

template < mesh::ElementType ET, el_o_t EO, size_t max_dofs_per_node, size_t num_maps, CondensationPolicy CP >
auto getUnsortedPrimaryDofs(const mesh::Element< ET, EO >&                                element,
                            const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& node_dof_map,
                            CondensationPolicyTag< CP >)
{
    return getDofsFromNodes(getPrimaryNodesArray< CP >(element), node_dof_map);
}
} // namespace lstr::dofs
#endif // L3STER_DOFS_DOFSFROMNODES_HPP
