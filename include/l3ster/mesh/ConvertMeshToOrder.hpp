#ifndef L3STER_MESH_CONVERTMESHTOORDER_HPP
#define L3STER_MESH_CONVERTMESHTOORDER_HPP

#include "l3ster/mesh/ElementIntersecting.hpp"
#include "l3ster/mesh/MeshUtils.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/Common.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/MetisUtils.hpp"

namespace lstr::mesh
{
namespace detail
{
struct NeighborInfo
{
    element_cptr_variant_t< 1 > ptr;
    d_id_t                      domain;
};

inline auto makeNeighborInfoMap(const MeshPartition< 1 >& mesh)
    -> robin_hood::unordered_flat_map< el_id_t, NeighborInfo >
{
    auto retval = robin_hood::unordered_flat_map< el_id_t, NeighborInfo >(mesh.getNElements());
    for (d_id_t domain_id : mesh.getDomainIds())
    {
        const auto insert_el = [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >& element) {
            const auto map_entry = NeighborInfo{&element, domain_id};
            retval.emplace(element.getId(), map_entry);
        };
        mesh.visit(insert_el, std::views::single(domain_id));
    }
    return retval;
}
} // namespace detail

template < el_o_t OC >
auto convertMeshToOrder(const MeshPartition< 1 >& mesh, std::integral_constant< el_o_t, OC > = {})
    -> MeshPartition< OC >
{
    L3STER_PROFILE_FUNCTION;
    const auto nbr_info_map = detail::makeNeighborInfoMap(mesh);
    const auto dual_graph   = computeMeshDual(mesh);
    auto       new_domains  = mesh.getConversionAlloc< OC >();
    n_id_t     max_node     = mesh.getOwnedNodes().size();
    auto       converted    = util::DynamicBitset{mesh.getNElements()};

    const auto convert_domain = [&](const Domain< 1 >& old_domain, Domain< OC >& new_domain) {
        const auto convert_element = [&]< ElementType T, el_o_t O >(const Element< T, O >& el) {
            constexpr size_t n_new_nodes = Element< T, OC >::n_nodes;
            auto             mask        = std::bitset< n_new_nodes >{};
            auto             new_nodes   = std::array< n_id_t, n_new_nodes >{};
            updateMatchMask< OC >(el, mask, new_nodes);

            const auto match_nbr_nodes = [&](el_id_t nbr_id) {
                if (not converted.test(nbr_id))
                    return;

                const auto [nbr_ptr_var, nbr_dom_id] = nbr_info_map.at(nbr_id);
                const auto update_from_nbr           = [&]< ElementType T_Nbr >(const Element< T_Nbr, 1 >* nbr_ptr) {
                    const auto& nbr_vec       = new_domains.at(nbr_dom_id).template getElementVector< T_Nbr, OC >();
                    const auto  get_id        = &Element< T_Nbr, OC >::getId;
                    const auto& converted_nbr = *std::ranges::lower_bound(nbr_vec, nbr_id, {}, get_id);
                    updateMatchMask(*nbr_ptr, converted_nbr, el, mask, new_nodes);
                };
                std::visit(update_from_nbr, nbr_ptr_var);
            };
            std::ranges::for_each(dual_graph.getElementAdjacent(el.getId()), match_nbr_nodes);

            for (size_t i = 0; i != mask.size(); ++i)
                if (not mask[i])
                    new_nodes[i] = max_node++;

            // This relies on the fact that we're iterating over domain elements of a given type in ascending ID order
            auto& new_elvec = new_domain.template getElementVector< T, OC >();
            new_elvec.emplace_back(new_nodes, ElementData< T, OC >{el.getData()}, el.getId());
            converted.set(el.getId());
        };
        old_domain.visit(convert_element, std::execution::seq);
    };

    for (auto domain_id : mesh.getDomainIds())
        convert_domain(mesh.getDomain(domain_id), new_domains[domain_id]);

    return {std::move(new_domains), util::makeIndexVector(max_node), max_node};
}
} // namespace lstr::mesh
#endif // L3STER_MESH_CONVERTMESHTOORDER_HPP
