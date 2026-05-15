#ifndef L3STER_MESH_CONVERTMESHTOORDER_HPP
#define L3STER_MESH_CONVERTMESHTOORDER_HPP

#include "l3ster/mesh/MeshUtils.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/SpatialHashTable.hpp"

namespace lstr::mesh
{
namespace detail
{
class ExtractedFeatures
{
    template < ElementType T, el_o_t O >
    struct ComputeMaxSideNodes
    {
        static constexpr auto value = std::ranges::max(ElementTraits< Element< T, O > >::boundary_table |
                                                       std::views::transform([](const auto& sn) { return sn.size(); }));
    };

public:
    static constexpr size_t max_feature_nodes = meta_transform_reduce< 0uz, ComputeMaxSideNodes, util::Max, 1 >;
    using feature_nodes_t                     = util::StaticVector< n_id_t, max_feature_nodes >;

private:
    struct NodeHash
    {
        static auto operator()(const feature_nodes_t& nodes)
        {
            return robin_hood::hash_bytes(nodes.data(), std::span{nodes}.size_bytes());
        }
    };

    using node_to_feature_map_t = robin_hood::unordered_map< feature_nodes_t, size_t, NodeHash >;

public:
    struct FeatureDescription
    {
        el_id_t     parent;
        el_locind_t num_new_nodes;
        bool        is_shared = false;
    };

    node_to_feature_map_t                                            node_to_feature_index_map;
    std::vector< FeatureDescription >                                features;
    robin_hood::unordered_flat_map< el_id_t, std::vector< size_t > > element_to_features;
};

// Pure faces - face internal nodes (no edges)
template < ElementType T, el_o_t O >
constexpr auto computePureFaces()
{
    using eltraits     = ElementTraits< Element< T, O > >;
    constexpr auto dim = eltraits::native_dim;
    if constexpr (dim == 1)
        return std::array< std::array< el_locind_t, 0 >, 0 >{};
    else if constexpr (dim == 2)
        return std::array< decltype(eltraits::internal_node_inds), 1 >{eltraits::internal_node_inds};
    else if constexpr (dim == 3)
    {
        static constexpr auto num_node_parent_faces = std::invoke([] {
            auto retval = std::array< std::uint8_t, eltraits::nodes_per_element >{};
            for (auto i : eltraits::boundary_table | std::views::join)
                ++retval[i];
            return retval;
        });
        constexpr auto        is_pure_face_node     = [](auto i) {
            return num_node_parent_faces[i] == 1;
        };
        constexpr auto num_face_nodes = std::invoke([] {
            return util::elwise(eltraits::boundary_table,
                                [](const auto& ni) { return std::ranges::count_if(ni, is_pure_face_node); });
        });

        auto face_offsets = std::array< size_t, num_face_nodes.size() + 1 >{};
        std::inclusive_scan(num_face_nodes.begin(), num_face_nodes.end(), std::next(face_offsets.begin()));

        constexpr auto num_face_nodes_total = std::ranges::fold_left(num_face_nodes, 0uz, std::plus{});
        auto           face_nodes           = std::array< el_locind_t, num_face_nodes_total >{};
        std::ranges::copy_if(eltraits::boundary_table | std::views::join, face_nodes.begin(), is_pure_face_node);

        return std::make_pair(face_offsets, face_nodes);
    }
}
template < ElementType T, el_o_t O >
inline constexpr auto pure_faces = computePureFaces< T, O >();
template < ElementType T, el_o_t O >
auto getPureFaceRanges()
{
    constexpr auto dim = ElementTraits< Element< T, O > >::native_dim;
    if constexpr (dim == 3)
        return pure_faces< T, O >.first | std::views::adjacent_transform< 2 >([](size_t begin, size_t end) {
                   return std::span{pure_faces< T, O >.second}.subspan(begin, end - begin);
               });
    else
        return pure_faces< T, O >;
}

// Pure edges - edge internal nodes (no vertices)
template < ElementType T, el_o_t O >
constexpr auto computePureEdges()
{
    using eltraits = ElementTraits< Element< T, O > >;
    if constexpr (eltraits::native_dim == 1)
        return std::array< decltype(eltraits::internal_node_inds), 1 >{eltraits::internal_node_inds};
    else
    {
        auto retval = std::array< std::array< el_locind_t, O - 1 >, eltraits::edge_table.size() >{};
        for (auto&& [full_edge, pure_edge] : std::views::zip(eltraits::edge_table, retval))
            std::ranges::copy_if(full_edge, pure_edge.begin(), [](auto node) {
                return not std::ranges::contains(eltraits::vertices, node);
            });
        return retval;
    }
}
template < ElementType T, el_o_t O >
inline constexpr auto pure_edges = computePureEdges< T, O >();

template < el_o_t O >
auto extractFeatures(const MeshPartition< 1 >& mesh) -> ExtractedFeatures
{
    L3STER_PROFILE_FUNCTION;
    auto retval                 = ExtractedFeatures{};
    auto& [n2f, features, el2f] = retval;
    const auto put_feature =
        [&]< size_t N >(const std::array< n_id_t, N >& nodes, el_locind_t num_new_nodes, el_id_t el_id) -> size_t {
        const auto new_feature_id = features.size();
        if (const auto [iter, new_inserted] = n2f.try_emplace(nodes, new_feature_id); new_inserted)
        {
            features.push_back({.parent = el_id, .num_new_nodes = num_new_nodes});
            return new_feature_id;
        }
        else
        {
            const auto feat_ind             = iter->second;
            features.at(feat_ind).is_shared = true;
            return feat_ind;
        }
    };

    const auto extract_features = [&]< ElementType T >(util::ConstexprValue< T >,
                                                       const typename Element< T, 1 >::node_array_t& nodes,
                                                       el_id_t                                       id) {
        using eltraits1 = ElementTraits< Element< T, 1 > >;
        using eltraitsO = ElementTraits< Element< T, O > >;

        auto my_feature_ids = std::vector< size_t >{};

        constexpr auto dim = eltraits1::native_dim;
        if constexpr (dim == 3)
        {
            my_feature_ids.reserve(19);

            // Volume, unique by definition, don't need to store it in n2f
            constexpr auto num_vol_nodes = eltraitsO::internal_node_inds.size();
            my_feature_ids.push_back(features.size());
            features.push_back({.parent = id, .num_new_nodes = num_vol_nodes});

            // Faces
            for (auto&& [f1_ni, fO_nodes] : std::views::zip(eltraits1::boundary_table, getPureFaceRanges< T, O >()))
            {
                const auto face1_nodes   = util::getSortedArray(util::arrayAtInds(nodes, f1_ni));
                const auto num_new_nodes = static_cast< el_locind_t >(fO_nodes.size());
                const auto feature_ind   = put_feature(face1_nodes, num_new_nodes, id);
                my_feature_ids.push_back(feature_ind);
            }
        }
        else if constexpr (dim == 2)
        {
            my_feature_ids.reserve(7);

            // Face
            constexpr auto num_faceO_nodes = eltraitsO::internal_node_inds.size();
            const auto     face1_nodes     = util::getSortedArray(nodes);
            const auto     face_ind        = put_feature(face1_nodes, num_faceO_nodes, id);
            my_feature_ids.push_back(face_ind);
        }

        // Edges can be handled uniformly
        constexpr el_locind_t num_edgeO_nodes = O - 1;
        for (const auto& edge1_node_inds : eltraits1::edge_table)
        {
            const auto edge1_nodes = util::getSortedArray(util::arrayAtInds(nodes, edge1_node_inds));
            const auto feature_ind = put_feature(edge1_nodes, num_edgeO_nodes, id);
            my_feature_ids.push_back(feature_ind);
        }

        el2f.emplace(id, std::move(my_feature_ids));
    };

    const auto process_element = [&]< ElementType T >(const Element< T, 1 >& element) {
        constexpr auto type_wrp = util::ConstexprValue< ElementTraits< Element< T, 1 > >::geom_type >{};
        extract_features(type_wrp, element.nodes, element.id);
    };

    mesh.visit(process_element);
    return retval;
}

inline auto computeFeatureNodeStarts(const ExtractedFeatures& features, n_id_t o1_nodes)
{
    const auto&    f           = features.features;
    auto           retval      = util::ArrayOwner< n_id_t >(f.size());
    constexpr auto get_num_new = [](const auto& feat) -> n_id_t {
        return feat.num_new_nodes;
    };
    std::transform_exclusive_scan(f.begin(), f.end(), retval.begin(), o1_nodes, std::plus{}, get_num_new);
    return retval;
}

template < el_o_t O >
auto computeFeatureNodeCoords(const MeshPartition< 1 >& mesh, const ExtractedFeatures& features)
{
    constexpr auto get_num_new = [](const ExtractedFeatures::FeatureDescription& feat) -> n_id_t {
        return feat.is_shared ? feat.num_new_nodes : 0uz;
    };
    auto retval = util::CrsGraph< Point< 3 > >{features.features | std::views::transform(get_num_new)};

    const auto compute_feat_node_coords = [&]< ElementType T >(const Element< T, 1 >& element) {
        const auto& feature_inds       = features.element_to_features.at(element.id);
        const auto  synth_converted_el = Element< T, O >{{}, element.data, {}};
        size_t      i                  = 0;
        const auto  process_feature    = [&](std::span< const el_locind_t > node_inds) {
            const auto  feature_ind = feature_inds.at(i++);
            const auto& feature     = features.features.at(feature_ind);
            if (feature.parent == element.id and feature.is_shared)
                for (auto&& [node_i, coord] : std::views::zip(node_inds, retval(feature_ind)))
                    coord = nodePhysicalLocation(synth_converted_el, node_i);
        };

        // Ignore volume - no need to compute coords (volume is always uniquely owned)
        if constexpr (ElementTraits< Element< T, 1 > >::native_dim == 3)
            ++i;

        // Faces
        for (auto&& face_inds : getPureFaceRanges< T, O >())
            process_feature(face_inds);

        // Edges
        for (const auto& edge_inds : pure_edges< T, O >)
            process_feature(edge_inds);
    };

    mesh.visit(compute_feat_node_coords, std::execution::par);
    return retval;
}

template < el_o_t O >
auto initNewDomains(const MeshPartition< 1 >& mesh_old) -> MeshPartition< O >::domain_map_t
{
    auto retval = typename MeshPartition< O >::domain_map_t{};
    for (d_id_t domain_id : mesh_old.getDomainIds())
    {
        const auto& old_domain = mesh_old.getDomain(domain_id);
        auto&       new_domain = retval[domain_id];
        new_domain.dim         = old_domain.dim;
        new_domain.elements.resize(old_domain.elements.sizes());
    }
    return retval;
}

inline auto matchNode(std::span< const Point< 3 > > points, Point< 3 > node_coords, n_id_t node_start)
{
    constexpr auto match_points = [](const Point< 3 >& a, const Point< 3 >& b) {
        constexpr auto tolerance = 1.e-12;
        const auto     diffs     = util::elwise(a.coords, b.coords, std::minus{});
        const auto     bound_lo  = util::elwise(diffs, std::bind_back(std::greater{}, -tolerance));
        const auto     bound_hi  = util::elwise(diffs, std::bind_back(std::less{}, tolerance));
        return util::reduce(util::elwise(bound_lo, bound_hi, std::logical_and{}), std::logical_and{}, true);
    };
    util::throwingAssert(not points.empty());
    const auto match_iter = std::ranges::find_if(points, std::bind_back(match_points, node_coords));
    util::throwingAssert(match_iter != points.end());
    return node_start + static_cast< n_id_t >(std::distance(points.begin(), match_iter));
}

template < el_o_t O >
auto convertElements(const MeshPartition< 1 >&           mesh,
                     const ExtractedFeatures&            features,
                     const util::ArrayOwner< n_id_t >&   feature_node_starts,
                     const util::CrsGraph< Point< 3 > >& feature_node_coords) -> MeshPartition< O >
{
    const auto convert_element = [&]< ElementType T >(const Element< T, 1 >& element1) -> Element< T, O > {
        using eltraits1_t = ElementTraits< Element< T, 1 > >;
        using eltraitsO_t = ElementTraits< Element< T, O > >;

        auto retval            = Element< T, O >{{}, element1.data, element1.id};
        auto& [nodesO, _1, _2] = retval;

        const auto& feature_inds = features.element_to_features.at(element1.id);
        auto process_feature     = [&, fit = feature_inds.begin()](std::span< const el_locind_t > node_inds) mutable {
            const auto feat_ind   = *fit++;
            const auto node_start = feature_node_starts.at(feat_ind);
            const auto parent     = features.features.at(feat_ind).parent;
            if (parent == element1.id)
                for (auto [ni, n] : std::views::zip(node_inds, std::views::iota(node_start)))
                    nodesO[ni] = n;
            else
            {
                const auto feature_coords = feature_node_coords(feat_ind);
                for (auto ni : node_inds)
                {
                    const auto node_coords = nodePhysicalLocation(retval, ni);
                    const auto new_node_id = matchNode(feature_coords, node_coords, node_start);
                    nodesO[ni]             = new_node_id;
                }
            }
        };

        // Volume
        if constexpr (eltraits1_t::native_dim == 3)
            process_feature(std::span{eltraitsO_t::internal_node_inds});

        // Faces
        for (auto face_inds : getPureFaceRanges< T, O >())
            process_feature(face_inds);

        // Edges
        for (const auto& edge_inds : pure_edges< T, O >)
            process_feature(std::span{edge_inds});

        // Vertices
        for (auto&& [v1i, vOi] : std::views::zip(eltraits1_t::vertices, eltraitsO_t::vertices))
            nodesO[vOi] = element1.nodes[v1i];

        return retval;
    };

    const auto domain_ids  = util::ArrayOwner{mesh.getDomainIds()};
    auto       new_domains = initNewDomains< O >(mesh);
    util::tbb::parallelFor(domain_ids, [&](d_id_t dom_id) {
        const auto& d1 = mesh.getDomain(dom_id);
        auto&       dO = new_domains.at(dom_id);
        d1.elements.visitVectors([&]< ElementType T >(const std::vector< Element< T, 1 > >& vec1) {
            auto& vecO = dO.elements.template getVector< Element< T, O > >();
            util::tbb::parallelTransform(vec1, vecO.begin(), convert_element);
        });
    });

    const n_id_t max_node = feature_node_starts.back() + features.features.back().num_new_nodes;
    return {std::move(new_domains), 0, max_node, mesh.getBoundaryIdsCopy()};
}
} // namespace detail

template < el_o_t O >
auto convertMeshToOrder(const MeshPartition< 1 >& mesh) -> MeshPartition< O >
{
    if constexpr (O == 1)
        return copy(mesh);

    const auto features            = detail::extractFeatures< O >(mesh);
    const auto feature_node_coords = detail::computeFeatureNodeCoords< O >(mesh, features);
    const auto feature_node_starts = detail::computeFeatureNodeStarts(features, mesh.getNNodes());
    return detail::convertElements< O >(mesh, features, feature_node_starts, feature_node_coords);
}
} // namespace lstr::mesh
#endif // L3STER_MESH_CONVERTMESHTOORDER_HPP
