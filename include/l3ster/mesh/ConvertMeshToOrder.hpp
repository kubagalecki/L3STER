#ifndef L3STER_MESH_CONVERTMESHTOORDER_HPP
#define L3STER_MESH_CONVERTMESHTOORDER_HPP

#include "l3ster/mesh/ElementIntersecting.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/Common.hpp"
#include "l3ster/util/MetisUtils.hpp"

namespace lstr
{
template < el_o_t O_CONV >
auto convertMeshToOrder(const MeshPartition< 1 >& mesh, std::integral_constant< el_o_t, O_CONV > = {})
    -> MeshPartition< O_CONV >
{
    L3STER_PROFILE_FUNCTION;
    util::throwingAssert(mesh.isDualGraphInitialized(),
                         "Initialize the dual graph of the mesh before converting it to a different order");

    const auto&         dual_graph  = mesh.getDualGraph();
    auto                new_domains = mesh.getConversionAlloc< O_CONV >();
    n_id_t              max_node    = mesh.getOwnedNodes().size();
    std::vector< bool > converted(mesh.getNElements(), false);

    const auto convert_domain = [&](const Domain< 1 >& old_domain, Domain< O_CONV >& new_domain) {
        old_domain.visit(
            [&]< ElementType T, el_o_t O >(const Element< T, O >& el) {
                constexpr size_t                  n_new_nodes = Element< T, O_CONV >::n_nodes;
                std::bitset< n_new_nodes >        mask{};
                std::array< n_id_t, n_new_nodes > new_nodes;
                updateMatchMask< O_CONV >(el, mask, new_nodes);

                const auto match_nbr_nodes = [&](el_id_t nbr_id) {
                    if (not converted[nbr_id])
                        return;
                    const auto [nbr_ptr_var, nbr_dom_id] = *mesh.find(nbr_id);
                    std::visit(
                        [&]< ElementType T_N, el_o_t O_N >(const Element< T_N, O_N >* nbr_ptr) {
                            const Element< T_N, O_CONV >& converted_nbr = *std::ranges::lower_bound(
                                new_domains.at(nbr_dom_id).template getElementVector< T_N, O_CONV >(),
                                nbr_id,
                                {},
                                [](const auto& e) { return e.getId(); });
                            updateMatchMask(*nbr_ptr, converted_nbr, el, mask, new_nodes);
                        },
                        nbr_ptr_var);
                };
                std::ranges::for_each(dual_graph.getElementAdjacent(el.getId()), match_nbr_nodes);
                for (size_t i = 0; auto& n : new_nodes)
                    if (not mask[i++])
                        n = max_node++;
                new_domain.template emplaceBack< T, O_CONV >(
                    new_nodes, ElementData< T, O_CONV >{el.getData()}, el.getId());
                converted[el.getId()] = true;
            },
            std::execution::seq);
    };

    for (auto domain_id : mesh.getDomainIds())
        convert_domain(mesh.getDomain(domain_id), new_domains[domain_id]);

    return {std::move(new_domains), util::consecutiveIndices(max_node), max_node};
}
} // namespace lstr
#endif // L3STER_MESH_CONVERTMESHTOORDER_HPP
