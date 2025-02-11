#ifndef L3STER_DOFS_DOFSFROMNODES_HPP
#define L3STER_DOFS_DOFSFROMNODES_HPP

#include "l3ster/dofs/NodeToDofMap.hpp"

namespace lstr::dofs
{
template < CondensationPolicy CP, size_t max_dofs_per_node, mesh::ElementType ET, el_o_t EO, std::integral I >
auto getDofsCopy(const NodeToGlobalDofMap< max_dofs_per_node >& node2dof,
                 const mesh::Element< ET, EO >&                 element,
                 std::span< const I >                           dof_inds,
                 CondensationPolicyTag< CP > = {})
{
    constexpr auto num_nodes = dofs::getNumPrimaryNodes< CP, ET, EO >();
    auto           retval    = util::StaticVector< global_dof_t, num_nodes * max_dofs_per_node >{};
    for (auto node : getPrimaryNodesView< CP >(element))
    {
        const auto& node_dofs = node2dof(node);
        for (auto i : dof_inds)
        {
            const auto dof = node_dofs[i];
            if (dof != invalid_global_dof)
                retval.push_back(dof);
        }
    }
    std::ranges::sort(retval);
    const auto erase_begin = std::ranges::unique(retval).begin();
    retval.erase(erase_begin, retval.end());
    return retval;
}

template < IndexRange_c auto dof_inds, size_t n_nodes, size_t dofs_per_node >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&             nodes,
                      const dofs::NodeToGlobalDofMap< dofs_per_node >& node_dof_map,
                      util::ConstexprValue< dof_inds >                 dofinds_ctwrpr = {})
    -> std::array< global_dof_t, std::ranges::size(dof_inds) * n_nodes >
{
    std::array< global_dof_t, std::ranges::size(dof_inds) * n_nodes > retval;
    std::ranges::copy(nodes | std::views::transform([&](n_id_t node) {
                          return util::getValuesAtInds(node_dof_map(node), dofinds_ctwrpr);
                      }) | std::views::join,
                      begin(retval));
    return retval;
}

template < size_t n_nodes, size_t dofs_per_node, SizedRangeOfConvertibleTo_c< size_t > Inds >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&             nodes,
                      const dofs::NodeToGlobalDofMap< dofs_per_node >& node_dof_map,
                      Inds&&                                           dof_inds)
{
    return nodes | std::views::transform([&](n_id_t node) {
               const auto& all_dofs = node_dof_map(node);
               return dof_inds | std::views::transform([&](size_t i) { return all_dofs[i]; });
           }) |
           std::views::join;
}

template < size_t n_nodes, size_t dofs_per_node >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&             nodes,
                      const dofs::NodeToGlobalDofMap< dofs_per_node >& node_dof_map)
    -> util::StaticVector< global_dof_t, dofs_per_node * n_nodes >
{
    util::StaticVector< global_dof_t, dofs_per_node * n_nodes > retval;
    std::ranges::remove_copy(nodes | std::views::transform([&](n_id_t node) { return node_dof_map(node); }) |
                                 std::views::join,
                             std::back_inserter(retval),
                             invalid_global_dof);
    return retval;
}

template < IndexRange_c auto dof_inds, size_t n_nodes, size_t dofs_per_node, size_t num_maps >
auto getDofsFromNodes(const std::array< n_id_t, n_nodes >&                      nodes,
                      const dofs::NodeToLocalDofMap< dofs_per_node, num_maps >& node_dof_map,
                      const util::ConstexprValue< dof_inds >                    dofinds_ctwrpr = {})
{
    using dof_array_t = std::array< local_dof_t, std::ranges::size(dof_inds) * n_nodes >;
    auto retval       = std::array< dof_array_t, num_maps >{};
    auto iters        = std::array< typename dof_array_t::iterator, num_maps >{};
    std::ranges::transform(retval, begin(iters), [](auto& arr) { return arr.begin(); });
    for (auto node : nodes)
    {
        const auto& all_dof_arrays = node_dof_map(node);
        for (auto&& [i, all_dofs] : all_dof_arrays | std::views::enumerate)
            iters[i] = util::copyValuesAtInds(all_dofs, iters[i], dofinds_ctwrpr);
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
        for (auto&& [i, all_dofs] : all_dof_arrays | std::views::enumerate)
            std::ranges::remove_copy(all_dofs, std::back_inserter(retval[i]), invalid_global_dof);
    }
    return retval;
}
} // namespace lstr::dofs
#endif // L3STER_DOFS_DOFSFROMNODES_HPP
