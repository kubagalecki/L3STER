#ifndef L3STER_MESHUTILS_HPP
#define L3STER_MESHUTILS_HPP

#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/mesh/NodeLocation.hpp"
#include "l3ster/util/Functional.hpp"
#include "l3ster/util/IndexMap.hpp"
#include "l3ster/util/Serialization.hpp"
#include "l3ster/util/SpatialHashTable.hpp"

namespace lstr::mesh
{
template < el_o_t... orders, ElementType BET, el_o_t BEO >
auto findDomainElement(const MeshPartition< orders... >& mesh,
                       const Element< BET, BEO >&        bnd_el,
                       const util::ArrayOwner< d_id_t >& domain_ids)
    -> std::optional< std::pair< element_cptr_variant_t< orders... >, el_side_t > >
{
    auto       retval              = std::optional< std::pair< element_cptr_variant_t< orders... >, el_side_t > >{};
    const auto bnd_el_nodes_sorted = util::getSortedArray(bnd_el.nodes);
    const auto match_domain_el     = [&]< ElementType DET, el_o_t DEO >(const Element< DET, DEO >& domain_el) {
        if constexpr (ElementTraits< Element< DET, DEO > >::native_dim ==
                      ElementTraits< Element< BET, BEO > >::native_dim + 1)
        {
            const auto matched_side = detail::matchBoundaryNodesToElement(domain_el, bnd_el_nodes_sorted);
            if (matched_side)
                retval.emplace(&domain_el, *matched_side);
            return matched_side.has_value();
        }
        else
            return false;
    };
    mesh.find(match_domain_el, domain_ids);
    return retval;
}

template < el_o_t... orders >
bool isUnpartitioned(const MeshPartition< orders... >& mesh)
{
    return mesh.getNNodes() == 0 or
           (mesh.getNodeOwnership().shared().empty() and
            mesh.getNodeOwnership().owned().back() + 1 == mesh.getNodeOwnership().owned().size());
}

struct MeshDualGraph
{
    util::CrsGraph< el_loc_id_t >          graph;          // adjacency graph
    util::CrsGraph< unsigned >             weights;        // weights
    util::ArrayOwner< el_id_t >            elements;       // element global IDs
    util::IndexMap< el_id_t, el_loc_id_t > els_gid_to_lid; // global-to-local element ID map
};
template < el_o_t... orders >
auto computeMeshDual(const MeshPartition< orders... >& mesh, size_t num_common_nodes = 1) -> MeshDualGraph
{
    auto element_ids = util::ArrayOwner< el_id_t >(mesh.getNElements(), std::numeric_limits< el_id_t >::max());
    auto i           = 0uz;
    mesh.visit([&](const auto& element) { element_ids[i++] = element.id; });
    std::ranges::sort(element_ids);
    auto        element_g2l      = util::IndexMap< el_id_t, el_loc_id_t >{element_ids};
    const auto& node_ownership   = mesh.getNodeOwnership();
    auto        node_degs        = util::ArrayOwner< unsigned >(node_ownership.localSize(), 0);
    const auto  update_node_degs = [&](const auto& element) {
        for (auto n : element.nodes)
            std::atomic_ref{node_degs[node_ownership.getLocalIndex(n)]}.fetch_add(1, std::memory_order_relaxed);
    };
    mesh.visit(update_node_degs, std::execution::par);
    auto       node2elems     = util::CrsGraph< el_loc_id_t >{node_degs};
    const auto write_elem_ids = [&](const auto& element) {
        const auto elid = element_g2l(element.id);
        for (auto n : element.nodes)
        {
            const auto nlid         = node_ownership.getLocalIndex(n);
            const auto index        = std::atomic_ref{node_degs[nlid]}.fetch_sub(1, std::memory_order_acq_rel) - 1u;
            node2elems(nlid)[index] = elid;
        }
    };
    mesh.visit(write_elem_ids, std::execution::par);
    auto       elem_degs               = util::ArrayOwner< unsigned >(element_ids.size(), 0);
    const auto get_neighbors_with_reps = [&](const auto& element, el_loc_id_t elid) {
        auto retval = element.nodes |
                      std::views::transform([&](auto node) { return node2elems(node_ownership.getLocalIndex(node)); }) |
                      std::views::join | std::views::filter(std::bind_back(std::not_equal_to{}, elid)) |
                      std::ranges::to< util::ArrayOwner >();
        std::ranges::sort(retval);
        return retval;
    };
    const auto update_elem_degs = [&](const auto& element) {
        const auto elid     = element_g2l(element.id);
        const auto nbrs     = get_neighbors_with_reps(element, elid);
        const auto num_nbrs = std::ranges::count_if(nbrs | std::views::chunk_by(std::equal_to{}), [&](auto&& reps) {
            return std::ranges::size(std::forward< decltype(reps) >(reps)) >= num_common_nodes;
        });
        elem_degs.at(elid)  = static_cast< unsigned >(num_nbrs);
    };
    mesh.visit(update_elem_degs, std::execution::par);
    auto       dual_graph       = util::CrsGraph< el_loc_id_t >{elem_degs};
    auto       weights          = util::CrsGraph< unsigned >{elem_degs};
    const auto write_graph_data = [&](const auto& element) {
        const auto elid       = element_g2l(element.id);
        const auto nbrs       = get_neighbors_with_reps(element, elid);
        const auto dest_verts = dual_graph(elid);
        const auto dest_wgts  = weights(elid);
        auto       view       = std::views::zip(
            dest_verts, dest_wgts, nbrs | std::views::chunk_by(std::equal_to{}) | std::views::filter([&](auto&& r) {
                                       return std::ranges::size(std::forward< decltype(r) >(r)) >= num_common_nodes;
                                   }));
        for (auto&& [v, w, reps] : view)
        {
            v = *std::ranges::begin(reps);
            w = static_cast< unsigned >(std::ranges::size(reps));
        }
    };
    mesh.visit(write_graph_data, std::execution::par);
    return {std::move(dual_graph), std::move(weights), std::move(element_ids), std::move(element_g2l)};
}

namespace detail
{
inline auto makeBoundaryNodeCoordsMap(const MeshPartition< 1 >& mesh)
    -> robin_hood::unordered_flat_map< n_id_t, Point< 3 > >
{
    constexpr auto nan         = std::numeric_limits< val_t >::quiet_NaN();
    constexpr auto nan_point   = Point< 3 >{nan, nan, nan};
    auto           node_lookup = util::ArrayOwner< Point< 3 > >(mesh.getNodeOwnership().localSize(), nan_point);
    const auto     put_nodes   = [&]< ElementType ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) {
        const auto& ref_x = mesh::getNodeLocations< ET, EO >();
        for (auto i : el_view.getSideNodeInds())
        {
            const auto  node_id       = el_view->nodes[i];
            const auto& ref_location  = ref_x[i];
            const auto  phys_location = map::mapToPhysicalSpace(el_view->data, ref_location);
            auto&       dest_xyz      = node_lookup.at(node_id);
            for (auto&& [x_src, x_dest] : std::views::zip(phys_location, dest_xyz))
                std::atomic_ref{x_dest}.store(x_src, std::memory_order_relaxed);
        }
    };
    mesh.visitBoundaries(put_nodes, mesh.getBoundaryIdsView(), std::execution::par);
    auto retval = robin_hood::unordered_flat_map< n_id_t, Point< 3 > >{};
    for (auto&& [node, coords] : node_lookup | std::views::enumerate)
        if (coords != nan_point)
            retval[static_cast< n_id_t >(node)] = coords;
    return retval;
}

inline auto initSpaceHashTable(const robin_hood::unordered_flat_map< n_id_t, Point< 3 > >& node_to_coord)
    -> util::SpatialHashTable< n_id_t, 3 >
{
    using util::elwise;
    constexpr auto inf    = std::numeric_limits< val_t >::infinity();
    constexpr auto init1  = std::array{inf, inf, inf};
    constexpr auto init   = std::array{init1, init1};
    using elem1_t         = std::array< val_t, 3 >;
    using elem_t          = std::array< elem1_t, 2 >;
    constexpr auto reduce = [](const elem_t& a, const elem_t& b) {
        const auto& [a1, a2] = a;
        const auto& [b1, b2] = b;
        return elem_t{elwise(a1, b1, util::Min{}), elwise(a2, b2, util::Min{})};
    };
    const auto coords_view    = node_to_coord | std::views::transform([](const auto& p) {
                                 const auto coord_array = static_cast< elem1_t >(p.second);
                                 return elem_t{coord_array, elwise(coord_array, std::negate{})};
                             });
    const auto [min, neg_max] = std::ranges::fold_left(coords_view, init, reduce);
    const auto max            = elwise(neg_max, std::negate{});
    const auto bb             = elwise(max, min, std::minus{});
    const bool is_2d          = bb.back() <= std::numeric_limits< val_t >::epsilon();
    const auto nn_fp          = static_cast< val_t >(node_to_coord.size());
    const auto nodes_1d_est   = is_2d ? nn_fp : std::sqrt(nn_fp);
    const auto grid_step      = elwise(bb, std::bind_front(std::multiplies{}, 1. / (100. * nodes_1d_est)));
    const auto origin         = elwise(elwise(min, max, std::plus{}), std::bind_front(std::multiplies{}, .5));
    return util::SpatialHashTable< n_id_t, 3 >{grid_step, origin};
}

inline auto invertNodeCoordMap(const robin_hood::unordered_flat_map< n_id_t, Point< 3 > >& node_to_coord)
    -> util::SpatialHashTable< n_id_t, 3 >
{
    auto retval = initSpaceHashTable(node_to_coord);
    for (const auto& [node, coords] : node_to_coord)
        retval.insert(coords, node);
    return retval;
}

inline auto matchExistingNodes(const util::SpatialHashTable< n_id_t, 3 >&                  existing,
                               const robin_hood::unordered_flat_map< n_id_t, Point< 3 > >& to_match)
    -> robin_hood::unordered_flat_map< n_id_t, n_id_t >
{
    constexpr auto match = [](const auto& p1, const auto& map_entry) {
        using namespace util;
        constexpr auto eps = 1e-12;
        const auto&    p2  = map_entry.first;
        return std::sqrt(reduce(elwise(elwise(p1, p2, std::minus{}), selfie(std::multiplies{})), std::plus{})) < eps;
    };
    auto retval = robin_hood::unordered_flat_map< n_id_t, n_id_t >{};
    for (const auto& [node, coord] : to_match)
    {
        const auto& coord_array      = static_cast< std::array< val_t, 3 > >(coord);
        auto        possible_matches = existing.proximate(coord);
        const auto  found            = std::ranges::find_if(possible_matches, std::bind_front(match, coord_array));
        if (found != possible_matches.end())
            retval[node] = found->second;
    }
    return retval;
}

inline auto getFaceSet(const MeshPartition< 1 >& mesh)
{
    constexpr auto hash = [](const util::ArrayOwner< n_id_t >& face_node) {
        const auto span = std::span{face_node};
        return robin_hood::hash_bytes(span.data(), span.size_bytes());
    };
    auto       retval   = robin_hood::unordered_flat_set< util::ArrayOwner< n_id_t >, decltype(hash) >{};
    const auto put_face = [&]< ElementType ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) {
        auto face_nodes = util::ArrayOwner{el_view.getSideNodesView()};
        std::ranges::sort(face_nodes);
        retval.insert(std::move(face_nodes));
    };
    mesh.visitBoundaries(put_face, mesh.getBoundaryIdsView(), std::execution::seq);
    return retval;
}
using FaceSet = std::invoke_result_t< decltype(&getFaceSet), const MeshPartition< 1 >& >;

inline auto getFacesToDelete(const FaceSet&                                          existing,
                             const FaceSet&                                          incoming,
                             const robin_hood::unordered_flat_map< n_id_t, n_id_t >& inc_to_ex_map)
    -> std::array< FaceSet, 2 >
{
    using face_t                             = util::ArrayOwner< n_id_t >;
    auto retval                              = std::array< FaceSet, 2 >{};
    auto& [existing_to_del, incoming_to_del] = retval;
    const auto check_face                    = [&](const face_t& face_incoming) {
        auto face_existing = face_t(face_incoming.size());
        for (auto&& [inc, ex] : std::views::zip(face_incoming, face_existing))
        {
            const auto iter = inc_to_ex_map.find(inc);
            if (iter == inc_to_ex_map.end())
                return;
            ex = iter->second;
        }
        std::ranges::sort(face_existing);
        if (existing.contains(face_existing))
        {
            existing_to_del.insert(std::move(face_existing));
            incoming_to_del.insert(copy(face_incoming));
        }
    };
    for (const auto& face : incoming)
        check_face(face);
    return retval;
}

template < typename ElementProj >
auto copyElements(MeshPartition< 1 >::domain_map_t& domains,
                  const MeshPartition< 1 >&         mesh,
                  const FaceSet&                    faces_to_delete,
                  ElementProj&&                     el_proj) -> std::vector< el_id_t >
{
    const auto is_deleted = [&](const auto& element) {
        auto nodes = std::vector(element.nodes.begin(), element.nodes.end());
        std::ranges::sort(nodes);
        return faces_to_delete.contains(nodes);
    };
    auto domain_ids = std::vector< d_id_t >{};
    std::ranges::set_difference(mesh.getDomainIds(), mesh.getBoundaryIdsView(), std::back_inserter(domain_ids));
    for (auto domain_id : domain_ids)
    {
        auto& domain = domains[domain_id];
        mesh.visit([&](const auto& element) { pushToDomain(domain, std::invoke(el_proj, element)); }, domain_id);
    }
    auto deleted_ids = std::vector< el_id_t >{};
    deleted_ids.reserve(faces_to_delete.size());
    for (auto domain_id : mesh.getBoundaryIdsView())
    {
        auto& domain = domains[domain_id];
        mesh.visit(
            [&](const auto& element) {
                const auto new_element = std::invoke(el_proj, element);
                if (is_deleted(element))
                    deleted_ids.push_back(new_element.id);
                else
                    pushToDomain(domain, new_element);
            },
            domain_id,
            std::execution::seq);
        if (domain.elements.empty())
            domains.erase(domains.find(domain_id));
    }
    std::ranges::sort(deleted_ids);
    return deleted_ids;
}

inline void fixElementIds(MeshPartition< 1 >::domain_map_t& domains, std::span< const el_id_t > deleted_ids)
{
    const auto get_updated_id = [&](el_id_t id) {
        const auto lb_iter            = std::ranges::lower_bound(deleted_ids, id);
        const auto num_lesser_deleted = static_cast< el_id_t >(std::distance(deleted_ids.begin(), lb_iter));
        return id - num_lesser_deleted;
    };
    const auto domain_ids = util::ArrayOwner{domains | std::views::keys};
    util::tbb::parallelFor(domain_ids, [&](d_id_t domain_id) {
        auto& domain = domains.at(domain_id);
        domain.elements.visit(
            [&]< ElementType ET, el_o_t EO >(Element< ET, EO >& element) { element.id = get_updated_id(element.id); },
            std::execution::par);
        domain.elements.visitVectors([]< ElementType ET, el_o_t EO >(std::vector< Element< ET, EO > >& elements) {
            std::ranges::sort(elements, {}, &Element< ET, EO >::id);
        });
    });
}

template < ElementType ET, el_o_t EO >
auto translateElement(const Element< ET, EO >&                                element,
                      const robin_hood::unordered_flat_map< n_id_t, n_id_t >& node_map,
                      n_id_t                                                  num_existing_nodes,
                      el_id_t                                                 num_existing_els) -> Element< ET, EO >
{
    auto renumbered = util::ArrayOwner{node_map | std::views::transform([](const auto& p) { return p.first; })};
    std::ranges::sort(renumbered);
    const auto new_node_id = [&](n_id_t id) {
        if (const auto renum_iter = node_map.find(id); renum_iter != node_map.end())
            return renum_iter->second;
        const auto lb               = std::ranges::lower_bound(renumbered, id);
        const auto num_lesser_renum = static_cast< n_id_t >(std::distance(renumbered.begin(), lb));
        return id + num_existing_nodes - num_lesser_renum;
    };
    auto new_nodes = typename Element< ET, EO >::node_array_t{};
    std::ranges::transform(element.nodes, new_nodes.begin(), new_node_id);
    return {new_nodes, element.data, element.id + num_existing_els};
}

template < el_o_t... EO >
using tuple_of_elarrays_t = std::invoke_result_t< decltype([]< typename... Ts >(const util::UniVector< Ts... >&) {
                                                      return std::tuple< util::ArrayOwner< Ts >... >{};
                                                  }),
                                                  typename Domain< EO... >::el_univec_t >;
} // namespace detail

template < el_o_t... EO >
auto serializeMesh(const MeshPartition< EO... >& mesh) -> std::string
{
    const auto  domain_ids      = mesh.getDomainIds();
    const auto  elem_data       = domain_ids | std::views::transform([&](auto d_id) {
                               const auto el_spans = getSpans(mesh.getDomain(d_id).elements);
                               return std::make_pair(d_id, el_spans);
                           });
    const auto  num_owned_nodes = mesh.getNodeOwnership().owned().size();
    const auto  nodes_begin     = num_owned_nodes ? mesh.getNodeOwnership().owned().front() : n_id_t{0};
    const auto  bnd_ids         = mesh.getBoundaryIdsView();
    std::string retval;
    util::serialize(std::make_tuple(elem_data, nodes_begin, num_owned_nodes, bnd_ids), std::back_inserter(retval));
    return retval;
}

template < el_o_t... EO >
auto deserializeMesh(std::string_view serial) -> MeshPartition< EO... >
{
    using domain_descr_t = std::pair< d_id_t, detail::tuple_of_elarrays_t< EO... > >;
    using elem_data_t    = util::ArrayOwner< domain_descr_t >;
    using deserial_t     = std::tuple< elem_data_t, n_id_t, size_t, util::ArrayOwner< d_id_t > >;
    const auto [elem_data, nodes_begin, num_owned_nodes, bnd_ids] = util::deserialize< deserial_t >(serial);

    constexpr auto make_domain = [](const domain_descr_t& dom_descr) {
        auto       domain        = Domain< EO... >{};
        const auto copy_elements = [&]< typename... elem_ts >(const util::ArrayOwner< elem_ts >&... elem_arrays) {
            const auto sizes = std::array{elem_arrays.size()...};
            domain.elements.reserve(sizes);
            const auto copy_elems = [&]< typename el_t >(const util::ArrayOwner< el_t >& elem_array) {
                for (const auto& el : elem_array)
                    pushToDomain(domain, el);
            };
            (copy_elems(elem_arrays), ...);
        };
        std::apply(copy_elements, dom_descr.second);
        return std::make_pair(dom_descr.first, std::move(domain));
    };
    using dom_map_t = MeshPartition< EO... >::domain_map_t;
    auto domain_map = elem_data | std::views::transform(make_domain) | std::ranges::to< dom_map_t >();

    return {std::move(domain_map), nodes_begin, num_owned_nodes, bnd_ids};
}

inline auto merge(const MeshPartition< 1 >& mesh1, const MeshPartition< 1 >& mesh2) -> MeshPartition< 1 >
{
    const auto node_to_coord1     = detail::makeBoundaryNodeCoordsMap(mesh1);
    const auto node_to_coord2     = detail::makeBoundaryNodeCoordsMap(mesh2);
    const auto coord_to_node1     = detail::invertNodeCoordMap(node_to_coord1);
    const auto faces1             = detail::getFaceSet(mesh1);
    const auto faces2             = detail::getFaceSet(mesh2);
    const auto match_map          = detail::matchExistingNodes(coord_to_node1, node_to_coord2);
    const auto [to_del1, to_del2] = detail::getFacesToDelete(faces1, faces2, match_map);
    auto       domains            = MeshPartition< 1 >::domain_map_t{};
    const auto deleted_ids1       = detail::copyElements(domains, mesh1, to_del1, std::identity{});
    const auto deleted_ids2       = detail::copyElements(domains, mesh2, to_del2, [&](const auto& el) {
        return detail::translateElement(el, match_map, mesh1.getNNodes(), mesh1.getNElements());
    });
    const auto deleted_ids        = util::concatRanges(deleted_ids1, deleted_ids2);
    detail::fixElementIds(domains, deleted_ids);
    const n_id_t num_nodes    = mesh1.getNNodes() + mesh2.getNNodes() - match_map.size();
    auto         boundary_ids = util::concatRanges(mesh1.getBoundaryIdsView(), mesh2.getBoundaryIdsView());
    boundary_ids              = util::getUniqueCopy(std::move(boundary_ids));
    auto remaining_boundaries = std::vector< d_id_t >{};
    std::ranges::set_intersection(boundary_ids, domains | std::views::keys, std::back_inserter(remaining_boundaries));
    return {domains, 0, num_nodes, remaining_boundaries};
}

template < Mapping_c< Point< 3 >, Point< 3 > > Deformation >
auto deform(MeshPartition< 1 >& mesh, Deformation&& deform)
{
    mesh.visit(
        [&]< ElementType ET, el_o_t EO >(Element< ET, EO >& element) {
            for (auto& point : element.data.vertices)
                point = std::invoke(deform, point);
        },
        std::execution::par);
}

template < std::ranges::random_access_range R >
auto extrude(const MeshPartition< 1 >& mesh, R&& zdist, d_id_t id_back, d_id_t id_front) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< R > >, val_t >
{
    const auto n_layers = std::ranges::distance(zdist);
    util::throwingAssert(n_layers > 1);
    auto       domains3d       = MeshPartition< 1 >::domain_map_t{};
    const auto nodes_per_layer = mesh.getNNodes();
    const auto z_min           = *std::ranges::begin(zdist);
    const auto z_max           = *std::ranges::begin(zdist | std::views::reverse);
    auto       element_id      = el_id_t{0};

    const auto extrude_element = [&]< ElementType ET >(const Element< ET, 1 >&               element,
                                                       std::reference_wrapper< Domain< 1 > > domain) {
        constexpr auto nat_dim = Element< ET, 1 >::native_dim;
        util::throwingAssert(nat_dim < 3, "Cannot extrude 3D mesh");
        if constexpr (nat_dim < 3)
        {
            constexpr auto   ET_extruded = std::invoke([] {
                using enum ElementType;
                switch (ET)
                {
                case Line:
                    return Quad;
                case Line2:
                    return Quad2;
                case Quad:
                    return Hex;
                case Quad2:
                    return Hex2;
                }
            });
            constexpr size_t geom_order  = ElementTraits< Element< ET, 1 > >::geom_order;
            constexpr size_t n_verts2d   = ElementData< ET, 1 >::n_verts;
            constexpr size_t n_nodes2d   = Element< ET, 1 >::n_nodes;
            auto             data        = ElementData< ET_extruded, 1 >{};
            for (auto&& dest_verts : data.vertices | std::views::chunk(n_verts2d))
                std::ranges::copy(element.data.vertices, dest_verts.begin());
            for (auto&& [layer, zs] : zdist | std::views::adjacent< 2 > | std::views::enumerate)
            {
                const auto& [z_lo, z_hi] = zs;
                static_assert(geom_order <= 2, "[futureproof] Use Lobatto distribution for higher orders");
                const auto z_geom_pos = util::linspaceArray< geom_order + 1 >(z_lo, z_hi);
                for (auto&& [z, dest_verts] : std::views::zip(z_geom_pos, data.vertices | std::views::chunk(n_verts2d)))
                    for (auto& vertex : dest_verts)
                        vertex.z() = z;
                auto       nodes       = typename Element< ET_extruded, 1 >::node_array_t{};
                const auto node_offset = layer * nodes_per_layer;
                std::ranges::transform(element.nodes, nodes.begin(), std::bind_back(std::plus{}, node_offset));
                std::ranges::transform(element.nodes,
                                       std::next(nodes.begin(), n_nodes2d),
                                       std::bind_back(std::plus{}, node_offset + nodes_per_layer));
                auto el_extruded = Element< ET_extruded, 1 >{nodes, data, element_id++};
                pushToDomain(domain.get(), std::move(el_extruded));
            }
        }
    };
    const auto make_back_front_elems = [&]< ElementType ET >(const Element< ET, 1 >& element) {
        constexpr auto nat_dim = Element< ET, 1 >::native_dim;
        if constexpr (nat_dim == 2)
        {
            const auto make_face = [&](val_t z, d_id_t domain_id, n_id_t node_offs) {
                auto face = element;
                for (auto& n : face.nodes)
                    n += node_offs;
                for (auto& vertex : face.data.vertices)
                    vertex.z() = z;
                face.id = element_id++;
                pushToDomain(domains3d[domain_id], face);
            };
            make_face(z_min, id_back, 0);
            make_face(z_max, id_front, nodes_per_layer * (n_layers - 1u));
        }
    };

    for (auto domain_id : mesh.getDomainIds())
    {
        auto domain_ref = std::ref(domains3d[domain_id]);
        mesh.visit(std::bind_back(extrude_element, domain_ref), domain_id);
        mesh.visit(make_back_front_elems, domain_id);
    }

    return {domains3d, util::concatRanges(mesh.getBoundaryIdsView(), std::array{id_back, id_front})};
}
} // namespace lstr::mesh
#endif // L3STER_MESHUTILS_HPP
