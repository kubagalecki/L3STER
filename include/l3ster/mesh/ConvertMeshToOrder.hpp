#ifndef L3STER_MESH_CONVERTMESHTOORDER_HPP
#define L3STER_MESH_CONVERTMESHTOORDER_HPP

#include "l3ster/mesh/ElementIntersecting.hpp"
#include "l3ster/mesh/MeshUtils.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/DynamicBitset.hpp"

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
            retval.emplace(element.id, map_entry);
        };
        mesh.visit(insert_el, std::views::single(domain_id));
    }
    return retval;
}

template < el_o_t converted_order >
auto initNewDomains(const MeshPartition< 1 >& mesh_old) -> MeshPartition< converted_order >::domain_map_t
{
    auto retval = typename MeshPartition< converted_order >::domain_map_t{};
    for (d_id_t domain_id : mesh_old.getDomainIds())
    {
        const auto& old_domain = mesh_old.getDomain(domain_id);
        auto&       new_domain = retval[domain_id];
        new_domain.dim         = old_domain.dim;
        new_domain.elements.reserve(old_domain.elements.sizes());
    }
    return retval;
}
} // namespace detail

template < el_o_t OC >
auto convertMeshToOrder(const MeshPartition< 1 >& mesh, std::integral_constant< el_o_t, OC > = {})
    -> MeshPartition< OC >
{
    L3STER_PROFILE_FUNCTION;
    const auto nbr_info_map                            = detail::makeNeighborInfoMap(mesh);
    const auto [dual_graph, dual_wgts, el_ids, el_g2l] = computeMeshDual(mesh, 2);
    auto new_domains                                   = detail::initNewDomains< OC >(mesh);

    n_id_t max_node  = mesh.getNodeOwnership().owned().size();
    auto   converted = util::DynamicBitset{mesh.getNElements()};

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
                    const auto& univec        = new_domains.at(nbr_dom_id).elements;
                    const auto& nbr_vec       = univec.template getVector< Element< T_Nbr, OC > >();
                    const auto  get_id        = &Element< T_Nbr, OC >::id;
                    const auto& converted_nbr = *std::ranges::lower_bound(nbr_vec, nbr_id, {}, get_id);
                    updateMatchMask(*nbr_ptr, converted_nbr, el, mask, new_nodes);
                };
                std::visit(update_from_nbr, nbr_ptr_var);
            };
            std::ranges::for_each(dual_graph(el.id), match_nbr_nodes);

            for (auto i : ElementTraits< Element< T, OC > >::boundary_node_inds)
                if (not mask[i])
                    new_nodes[i] = max_node++;
            for (auto i : ElementTraits< Element< T, OC > >::internal_node_inds)
                if (not mask[i])
                    new_nodes[i] = max_node++;

            // This relies on the fact that we're iterating over domain elements of a given type in ascending ID order
            emplaceInDomain< T, OC >(new_domain, new_nodes, ElementData< T, OC >{el.data}, el.id);
            converted.set(el.id);
        };
        old_domain.elements.visit(convert_element, std::execution::seq);
    };

    for (auto domain_id : mesh.getDomainIds())
        convert_domain(mesh.getDomain(domain_id), new_domains[domain_id]);

    return {std::move(new_domains), 0, max_node, mesh.getBoundaryIdsCopy()};
}
} // namespace lstr::mesh
#endif // L3STER_MESH_CONVERTMESHTOORDER_HPP
