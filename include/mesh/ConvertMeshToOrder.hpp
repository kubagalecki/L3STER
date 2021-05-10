#ifndef L3STER_MESH_CONVERTMESHTOORDER_HPP
#define L3STER_MESH_CONVERTMESHTOORDER_HPP
#include "mesh/ElementIntersecting.hpp"
#include "mesh/Mesh.hpp"
#include "util/Common.hpp"
#include "util/MetisUtils.hpp"

namespace lstr
{
template < el_o_t O_C >
void convertMeshToOrder(Mesh& mesh)
{
    if (mesh.getPartitions().size() != 1)
        throw std::logic_error{"Cannot convert a mesh which is either empty or has been partitioned"};

    auto&       part       = mesh.getPartitions()[0];
    const auto& dual_graph = part.initDualGraph();

    auto                new_domains = part.getConversionAlloc< O_C >();
    n_id_t              max_node    = part.getNodes().size();
    std::vector< bool > converted(part.getNElements(), false);

    const auto convert_domain = [&](const Domain& old_domain, Domain& new_domain) {
        old_domain.cvisit([&]< ElementTypes T, el_o_t O >(const Element< T, O >& el) {
            if constexpr (O == 1)
            {
                constexpr size_t                  n_new_nodes = Element< T, O_C >::n_nodes;
                std::bitset< n_new_nodes >        mask{};
                std::array< n_id_t, n_new_nodes > new_nodes;

                const auto match_nbr_nodes = [&](el_id_t nbr_id) {
                    if (not converted[nbr_id])
                        return;
                    const auto [nbr_ptr_var, nbr_dom_id] = *part.find(nbr_id);
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
                new_domain.template emplaceBack< T, O_C >(new_nodes, ElementData< T, O_C >{el.getData()}, el.getId());
            }
        });
    };

    for (auto domain_id : part.getDomainIds())
        convert_domain(part.getDomain(domain_id), new_domains[domain_id]);

    part = MeshPartition{std::move(new_domains), consecutiveIndices(max_node), {}};
}
} // namespace lstr
#endif // L3STER_MESH_CONVERTMESHTOORDER_HPP
