#ifndef L3STER_MESHUTILS_HPP
#define L3STER_MESHUTILS_HPP

#include "l3ster/mesh/MeshPartition.hpp"

namespace lstr::mesh
{
namespace detail
{
template < el_side_t side, ElementType ET, el_o_t EO, size_t N >
constexpr bool doesSideMatch(const Element< ET, EO >& element, const std::array< n_id_t, N >& sorted_boundary_nodes)
{
    const auto&    side_node_inds = std::get< side >(ElementTraits< Element< ET, EO > >::boundary_table);
    constexpr auto n_side_nodes   = std::tuple_size_v< std::remove_cvref_t< decltype(side_node_inds) > >;
    if constexpr (n_side_nodes == N)
    {
        auto side_nodes = std::array< n_id_t, n_side_nodes >{};
        std::ranges::copy(util::makeIndexedView(element.getNodes(), side_node_inds), side_nodes.begin());
        std::ranges::sort(side_nodes);
        return std::ranges::equal(side_nodes, sorted_boundary_nodes);
    }
    else
        return false;
}
} // namespace detail

template < ElementType ET, el_o_t EO, size_t N >
constexpr auto matchBoundaryNodesToElement(const Element< ET, EO >&       element,
                                           const std::array< n_id_t, N >& sorted_boundary_nodes)
    -> std::optional< el_side_t >
{
    auto       matched_side   = el_side_t{};
    const auto fold_side_inds = [&]< el_side_t... sides >(std::integer_sequence< el_side_t, sides... >) {
        const auto match_side = [&]< el_side_t side >(std::integral_constant< el_side_t, side >) {
            if (detail::doesSideMatch< side >(element, sorted_boundary_nodes))
            {
                matched_side = side;
                return true;
            }
            else
                return false;
        };
        return (match_side(std::integral_constant< el_side_t, sides >{}) or ...);
    };
    constexpr auto side_inds = std::make_integer_sequence< el_side_t, ElementTraits< Element< ET, EO > >::n_sides >{};
    const auto     matched   = fold_side_inds(side_inds);
    if (matched)
        return {matched_side};
    else
        return std::nullopt;
}

template < el_o_t... orders >
auto makeDimToDomainMap(const MeshPartition< orders... >& mesh) -> std::map< dim_t, std::vector< d_id_t > >
{
    auto retval = std::map< dim_t, std::vector< d_id_t > >{};
    for (d_id_t id : mesh.getDomainIds())
    {
        const auto dom_dim = mesh.getDomainView(id).getDim();
        retval[dom_dim].push_back(id);
    }
    return retval;
}

template < el_o_t... orders, ElementType BET, el_o_t BEO >
auto findDomainElement(const MeshPartition< orders... >& mesh,
                       const Element< BET, BEO >&        bnd_el,
                       const util::ArrayOwner< d_id_t >& domain_ids)
    -> std::optional< std::pair< element_cptr_variant_t< orders... >, el_side_t > >
{
    auto       retval              = std::optional< std::pair< element_cptr_variant_t< orders... >, el_side_t > >{};
    auto       emplacement_mutex   = std::mutex{};
    const auto bnd_el_nodes_sorted = util::getSortedArray(bnd_el.getNodes());
    const auto match_domain_el     = [&]< ElementType DET, el_o_t DEO >(const Element< DET, DEO >& domain_el) {
        if constexpr (ElementTraits< Element< DET, DEO > >::native_dim ==
                      ElementTraits< Element< BET, BEO > >::native_dim + 1)
        {
            const auto matched_side = matchBoundaryNodesToElement(domain_el, bnd_el_nodes_sorted);
            if (matched_side)
            {
                const auto lock = std::lock_guard{emplacement_mutex}; // In a correct scenario there is no contention
                retval.emplace(&domain_el, *matched_side);
            }
            return matched_side.has_value();
        }
        else
            return false;
    };
    mesh.find(match_domain_el, domain_ids, std::execution::par);
    return retval;
}

template < el_o_t... orders >
bool isUnpartitioned(const MeshPartition< orders... >& mesh)
{
    return mesh.getAllNodes().empty() or
           (mesh.getGhostNodes().empty() and mesh.getOwnedNodes().back() + 1 == mesh.getOwnedNodes().size());
}

template < el_o_t... orders >
auto computeMeshDual(const MeshPartition< orders... >& mesh) -> util::metis::GraphWrapper
{
    util::throwingAssert(isUnpartitioned(mesh), "Adjacency graphs for partitioned meshes are not currently supported");

    constexpr std::string_view overflow_msg =
        "The mesh size exceeds the numeric limits of METIS' signed integer type. Continuing would "
        "result in signed integer overflow. Consider recompiling METIS with 64 bit integer support";
    constexpr auto max_metis_id = static_cast< std::uintmax_t >(std::numeric_limits< idx_t >::max());
    const auto     max_el_id    = static_cast< std::uintmax_t >(mesh.getNElements() + 1);
    const auto     max_n_id     = static_cast< std::uintmax_t >(mesh.getOwnedNodes().back());
    util::throwingAssert(max_el_id <= max_metis_id and max_n_id <= max_metis_id, overflow_msg);

    const auto convert_topo_to_metis_format = [&]() {
        const auto topo_size = std::invoke([&mesh]() {
            size_t retval = 0;
            mesh.visit([&retval](const auto& element) { retval += element.getNodes().size(); });
            return retval;
        });
        util::throwingAssert(static_cast< std::uintmax_t >(topo_size) <= max_metis_id, overflow_msg);

        auto retval        = std::array< std::vector< idx_t >, 2 >{};
        auto& [eptr, eind] = retval;
        eind.reserve(topo_size);
        eptr.reserve(mesh.getNElements() + 1);
        eptr.push_back(0);

        const auto convert_element = [&]< ElementType T, el_o_t O >(const Element< T, O >* element) {
            std::ranges::copy(element->getNodes(), std::back_inserter(eind));
            eptr.push_back(static_cast< idx_t >(eptr.back() + element->getNodes().size()));
        };
        for (el_id_t id = 0; id < mesh.getNElements(); ++id)
        {
            const auto el_ptr = mesh.find(id).value();
            std::visit(convert_element, el_ptr);
        };
        return retval;
    };
    auto [eptr, eind] = convert_topo_to_metis_format(); // should be const, but METIS API is const-averse
    idx_t  ne         = static_cast< idx_t >(mesh.getNElements());
    idx_t  nn         = static_cast< idx_t >(mesh.getOwnedNodes().size());
    idx_t  ncommon    = 2;
    idx_t  numflag    = 0;
    idx_t* xadj{};
    idx_t* adjncy{};

    const auto error_code = METIS_MeshToDual(&ne, &nn, eptr.data(), eind.data(), &ncommon, &numflag, &xadj, &adjncy);
    util::metis::handleMetisErrorCode(error_code);

    return util::metis::GraphWrapper{xadj, adjncy, mesh.getNElements()};
}
} // namespace lstr::mesh
#endif // L3STER_MESHUTILS_HPP
