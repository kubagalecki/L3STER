#ifndef L3STER_MESHUTILS_HPP
#define L3STER_MESHUTILS_HPP

#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"
#include "l3ster/util/Functional.hpp"
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

template < el_o_t... orders >
auto computeMeshDual(const MeshPartition< orders... >& mesh) -> util::metis::GraphWrapper
{
    util::throwingAssert(isUnpartitioned(mesh), "Adjacency graphs for partitioned meshes are not currently supported");

    constexpr std::string_view overflow_msg =
        "The mesh size exceeds the numeric limits of METIS' signed integer type. Continuing would "
        "result in signed integer overflow. Consider recompiling METIS with 64 bit integer support";
    constexpr auto max_metis_id = static_cast< std::uintmax_t >(std::numeric_limits< idx_t >::max());
    const auto     max_el_id    = static_cast< std::uintmax_t >(mesh.getNElements() + 1);
    const auto     max_n_id     = static_cast< std::uintmax_t >(mesh.getNodeOwnership().owned().back());
    util::throwingAssert(max_el_id <= max_metis_id and max_n_id <= max_metis_id, overflow_msg);

    const auto convert_topo_to_metis_format = [&]() {
        const auto topo_size = std::invoke([&mesh]() {
            size_t retval = 0;
            mesh.visit([&retval](const auto& element) { retval += element.nodes.size(); });
            return retval;
        });
        util::throwingAssert(static_cast< std::uintmax_t >(topo_size) <= max_metis_id, overflow_msg);

        auto retval        = std::array< std::vector< idx_t >, 2 >{};
        auto& [eptr, eind] = retval;
        eind.reserve(topo_size);
        eptr.reserve(mesh.getNElements() + 1);
        eptr.push_back(0);

        const auto convert_element = [&]< ElementType T, el_o_t O >(const Element< T, O >* element) {
            std::ranges::copy(element->nodes, std::back_inserter(eind));
            eptr.push_back(static_cast< idx_t >(eptr.back() + element->nodes.size()));
        };
        for (el_id_t id = 0; id < mesh.getNElements(); ++id)
        {
            const auto el_ptr = mesh.find(id).value();
            std::visit(convert_element, el_ptr);
        };
        return retval;
    };
    auto [eptr, eind] = convert_topo_to_metis_format(); // should be const, but METIS API is const-averse
    auto   ne         = static_cast< idx_t >(mesh.getNElements());
    auto   nn         = static_cast< idx_t >(mesh.getNodeOwnership().owned().size());
    idx_t  ncommon    = 2;
    idx_t  numflag    = 0;
    idx_t* xadj{};
    idx_t* adjncy{};

    const auto error_code = METIS_MeshToDual(&ne, &nn, eptr.data(), eind.data(), &ncommon, &numflag, &xadj, &adjncy);
    util::metis::handleMetisErrorCode(error_code);

    return util::metis::GraphWrapper{xadj, adjncy, mesh.getNElements()};
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
    const auto coords_view = node_to_coord | std::views::transform([](const auto& p) {
                                 const auto coord_array = static_cast< elem1_t >(p.second);
                                 return elem_t{coord_array, elwise(coord_array, std::negate{})};
                             }) |
                             std::views::common;
    const auto [min, neg_max] = std::reduce(coords_view.begin(), coords_view.end(), init, reduce);
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
        util::throwingAssert(Element< ET, 1 >::native_dim < 3, "Cannot extrude 3D mesh");
        if constexpr (ET == ElementType::Line or ET == ElementType::Quad)
        {
            constexpr auto   ET_extruded = ET == ElementType::Line ? ElementType::Quad : ElementType::Hex;
            constexpr size_t n_verts2d   = ElementData< ET, 1 >::n_verts;
            constexpr size_t n_nodes2d   = Element< ET, 1 >::n_nodes;
            auto             data        = ElementData< ET_extruded, 1 >{};
            std::ranges::copy(element.data.vertices, data.vertices.begin());
            std::ranges::copy(element.data.vertices, std::next(data.vertices.begin(), n_verts2d));
            for (auto&& [layer, zs] : zdist | std::views::adjacent< 2 > | std::views::enumerate)
            {
                const auto& [z_lo, z_hi] = zs;
                for (auto& vertex : data.vertices | std::views::take(n_verts2d))
                    vertex.z() = z_lo;
                for (auto& vertex : data.vertices | std::views::drop(n_verts2d))
                    vertex.z() = z_hi;
                auto nodes = typename Element< ET_extruded, 1 >::node_array_t{};
                std::ranges::transform(
                    element.nodes, nodes.begin(), std::bind_back(std::plus{}, layer * nodes_per_layer));
                std::ranges::transform(nodes | std::views::take(n_nodes2d),
                                       std::next(nodes.begin(), n_nodes2d),
                                       std::bind_back(std::plus{}, nodes_per_layer));
                auto el_extruded = Element< ET_extruded, 1 >{nodes, data, element_id++};
                pushToDomain(domain.get(), std::move(el_extruded));
            }
        }
    };
    const auto make_back_front_elems = [&]< ElementType ET >(const Element< ET, 1 >& element) {
        if constexpr (ET != ElementType::Line)
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
