#ifndef L3STER_MESH_CONVERTMESHTOORDER_HPP
#define L3STER_MESH_CONVERTMESHTOORDER_HPP

#include "l3ster/mesh/ElementIntersecting.hpp"
#include "l3ster/mesh/Mesh.hpp"
#include "l3ster/util/Common.hpp"
#include "l3ster/util/MetisUtils.hpp"

namespace lstr
{
template < el_o_t O_C >
[[nodiscard]] MeshPartition convertMeshToOrder(const MeshPartition& mesh)
{
    if (not mesh.isDualGraphInitialized())
        throw std::logic_error("Initialize the dual graph of the mesh before converting it to a different order");

    const auto&         dual_graph  = mesh.getDualGraph();
    auto                new_domains = mesh.getConversionAlloc< O_C >();
    n_id_t              max_node    = mesh.getNodes().size();
    std::vector< bool > converted(mesh.getNElements(), false);

    const auto convert_domain = [&](const Domain& old_domain, Domain& new_domain) {
        old_domain.visit(
            [&]< ElementTypes T, el_o_t O >(const Element< T, O >& el) {
                if constexpr (O == 1)
                {
                    constexpr size_t                  n_new_nodes = Element< T, O_C >::n_nodes;
                    std::bitset< n_new_nodes >        mask{};
                    std::array< n_id_t, n_new_nodes > new_nodes;
                    updateMatchMask< O_C >(el, mask, new_nodes);

                    const auto match_nbr_nodes = [&](el_id_t nbr_id) {
                        if (not converted[nbr_id])
                            return;
                        const auto [nbr_ptr_var, nbr_dom_id] = *mesh.find(nbr_id);
                        std::visit(
                            [&, ndi = nbr_dom_id]< ElementTypes T_N, el_o_t O_N >(const Element< T_N, O_N >* nbr_ptr) {
                                if constexpr (O_N == 1)
                                {
                                    const Element< T_N, O_C >& converted_nbr = *std::ranges::lower_bound(
                                        new_domains.at(ndi).template getElementVector< T_N, O_C >(),
                                        nbr_id,
                                        {},
                                        [](const auto& e) { return e.getId(); });
                                    updateMatchMask(*nbr_ptr, converted_nbr, el, mask, new_nodes);
                                }
                            },
                            nbr_ptr_var);
                    };
                    std::ranges::for_each(dual_graph.getElementAdjacent(el.getId()), match_nbr_nodes);
                    for (size_t i = 0; auto& n : new_nodes)
                        if (not mask[i++])
                            n = max_node++;
                    new_domain.template emplaceBack< T, O_C >(
                        new_nodes, ElementData< T, O_C >{el.getData()}, el.getId());
                    converted[el.getId()] = true;
                }
            },
            std::execution::seq);
    };

    for (auto domain_id : mesh.getDomainIds())
        convert_domain(mesh.getDomain(domain_id), new_domains[domain_id]);

    return MeshPartition{std::move(new_domains), consecutiveIndices(max_node), {}};
}
} // namespace lstr
#endif // L3STER_MESH_CONVERTMESHTOORDER_HPP
