#ifndef L3STER_MESH_READMESH_HPP
#define L3STER_MESH_READMESH_HPP

#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/IO.hpp"
#include "l3ster/util/Meta.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace lstr::mesh
{
enum class MeshFormat
{
    Gmsh,
};

// Disambiguation tag for reading meshes from different formats
template < MeshFormat >
struct MeshFormatTag
{};

constexpr inline MeshFormatTag gmsh_tag = MeshFormatTag< MeshFormat::Gmsh >{};

namespace detail
{
inline constexpr std::array gmsh_elt_lookup_tab{
    std::pair{1u, ElementType::Line}, std::pair{3u, ElementType::Quad}, std::pair{5u, ElementType::Hex}};

template < size_t I >
consteval ElementType lookupElt()
{
    return std::ranges::find(gmsh_elt_lookup_tab, I, [](const auto p) { return p.first; })->second;
}

struct NodeInfo
{
    n_id_t     l3ster_id;
    Point< 3 > coords;
};

template < ElementType ET >
auto makeElementFromNodeData(const std::array< NodeInfo, Element< ET, 1 >::n_nodes >& node_data, el_id_t id)
    -> Element< ET, 1 >
{
    using node_array_t   = std::array< n_id_t, Element< ET, 1 >::n_nodes >;
    using coords_array_t = ElementData< ET, 1 >::vertex_array_t;
    auto nodes           = node_array_t{};
    auto coords          = coords_array_t{};
    std::ranges::transform(node_data, nodes.begin(), [](const auto& n) { return n.l3ster_id; });
    std::ranges::transform(node_data, coords.begin(), [](const auto& n) { return n.coords; });

    if constexpr (ET == ElementType::Quad)
    {
        // L3STER node ordering is different from Gmsh's
        std::swap(nodes[2], nodes[3]);
        std::swap(coords[2], coords[3]);

        // In 2D, we need to ensure positive Jacobians. Gmsh may generate "upside down" elements, we need to flip those
        const auto same_z_as_first = [&coords](const Point< 3 >& p) {
            return std::fabs(p.z() - coords.front().z()) < 1e-9;
        };
        // We don't care about Quad elements residing in a general 3D space
        if (std::ranges::all_of(coords | std::views::drop(1), same_z_as_first))
        {
            // Check the face normal vector is pointing upwards (z+)
            const val_t ux = coords[1].x() - coords[0].x();
            const val_t uy = coords[1].y() - coords[0].y();
            const val_t vx = coords[2].x() - coords[0].x();
            const val_t vy = coords[2].y() - coords[0].y();
            const val_t nz = ux * vy - uy * vx;

            // Flip element
            if (nz < 0.)
            {
                std::swap(nodes[1], nodes[2]);
                std::swap(coords[1], coords[2]);
            }
        }
    }

    if constexpr (ET == ElementType::Hex)
    {
        // There is no special direction in 3D, we just need to permute the data in accordance with L3STER ordering
        std::swap(nodes[2], nodes[3]);
        std::swap(nodes[6], nodes[7]);
        std::swap(coords[2], coords[3]);
        std::swap(coords[6], coords[7]);
    }

    return Element< ET, 1 >{nodes, coords, id};
}
} // namespace detail

inline auto readMesh(std::string_view                  file_path,
                     const util::ArrayOwner< d_id_t >& boundary_ids,
                     MeshFormatTag< MeshFormat::Gmsh >) -> MeshPartition< 1 >
{
    L3STER_PROFILE_FUNCTION;

    auto file_map       = util::MmappedFile{std::string{file_path}};
    auto file_streambuf = util::MmappedStreambuf{std::move(file_map)};
    auto file           = std::istream{&file_streambuf};

    const auto assert_file_ok = [&](std::string_view     err_msg = "Error while reading .msh file",
                                    std::source_location sl      = std::source_location::current()) {
        util::throwingAssert(file.good(), err_msg, sl);
    };
    const auto skip_until_section = [&](std::string_view str) {
        const bool found = file_streambuf.skipPast(str);
        util::throwingAssert(found, "Error while reading .msh file: required section not present");
    };

    enum class Format
    {
        ASCII_V2,
        BIN32_V2,
        BIN64_V2,
        ASCII_V4,
        BIN32_V4,
        BIN64_V4,
    };
    const auto parse_format = [&]() -> Format {
        skip_until_section("$MeshFormat");

        float  version;
        bool   bin;
        size_t size;
        file >> version >> bin >> size;

        skip_until_section("$EndMeshFormat");

        Format format;
        if (version >= 4.0 and version < 5.0)
        {
            if (bin)
            {
                switch (size)
                {
                case 4:
                    format = Format::BIN32_V4;
                    break;
                case 8:
                    format = Format::BIN64_V4;
                    break;
                default:
                    util::throwingAssert(
                        false, "Error while reading .msh file: Unsupported size of size_t in the format section");
                }
            }
            else
                format = Format::ASCII_V4;
        }
        else
        {
            if (version >= 2.0 and version < 3.0)
            {
                if (bin)
                {
                    switch (size)
                    {
                    case 4:
                        format = Format::BIN32_V2;
                        break;
                    case 8:
                        format = Format::BIN64_V2;
                        break;
                    default:
                        util::throwingAssert(
                            false, "Error while reading .msh file: Unsupported size of size_t in the format section");
                    }
                }
                else
                    format = Format::ASCII_V2;
            }
            else
                util::throwingAssert(false, "Error while reading .msh file: Unsupported .msh format version");
        }

        util::throwingAssert(format == Format::ASCII_V4,
                             "Unsupported .msh format. Only the ASCII v4 gmsh format is currently supported");
        return format;
    };

    const auto parse_entities = [&](const Format&) {
        skip_until_section("$Entities");

        constexpr size_t n_entity_types = 4;
        using entity_data_t             = std::array< std::map< int, d_id_t >, n_entity_types >;
        auto entity_data                = entity_data_t{};

        const auto parse_entities_asciiv4 = [&]() {
            const auto n_entities = util::extract< size_t, 4 >(file);

            const auto parse_dim_entities = [&](const size_t dim) {
                const auto parse_entity = [&]() {
                    const int entity_tag = util::extract< int >(file);
                    util::ignore< val_t >(file, dim > 0 ? 6 : 3);

                    const auto n_physical_tags = util::extract< size_t >(file);
                    util::throwingAssert(n_physical_tags <= 1,
                                         "Error while reading .msh file: entity has more than 1 physical tag");
                    if (n_physical_tags == 1)
                        entity_data[dim][entity_tag] = util::extract< d_id_t >(file);

                    if (dim > 0)
                        util::ignore< int >(file, util::extract< size_t >(file));
                };

                for (size_t entity = 0; entity < n_entities[dim]; ++entity)
                    parse_entity();
            };

            for (size_t dim = 0; dim < n_entity_types; ++dim)
                parse_dim_entities(dim);

            skip_until_section("$EndEntities");
        };

        parse_entities_asciiv4();
        // TODO: switch over other possible formats once implemented

        const size_t n_physical_domains = std::invoke([&]() {
            std::set< d_id_t > unique_physical_ids;
            for (const auto& map : entity_data)
                for (const auto& [dim, entity] : map)
                    unique_physical_ids.insert(static_cast< d_id_t >(dim));
            return unique_physical_ids.size();
        });

        return std::make_pair(entity_data, n_physical_domains);
    };
    using entity_data_t = decltype(parse_entities(parse_format()));

    // Maps Gmsh ID to condensed ID + coordinates
    using node_data_t      = robin_hood::unordered_flat_map< size_t, detail::NodeInfo >;
    const auto parse_nodes = [&](const Format&) -> node_data_t {
        skip_until_section("$Nodes");

        const auto parse_nodes_asciiv4 = [&]() {
            const auto [n_blocks, n_nodes, min_node_tag, max_node_tag] = util::extract< size_t, 4 >(file);
            auto node_data                                             = node_data_t(n_nodes);

            auto   node_ids_in_block = std::vector< size_t >{};
            n_id_t l3ster_id         = 0;
            for (size_t block_ind = 0; block_ind < n_blocks; ++block_ind)
            {
                int    dim, entity_tag;
                bool   parametric;
                size_t block_size;
                file >> dim >> entity_tag >> parametric >> block_size;

                util::throwingAssert(!parametric, "Encountered parametric node while reading .msh file");

                node_ids_in_block.clear();
                node_ids_in_block.reserve(block_size);
                std::copy_n(std::istream_iterator< size_t >(file), block_size, std::back_inserter(node_ids_in_block));

                for (size_t gmsh_id : node_ids_in_block)
                {
                    auto coords = std::array< val_t, 3 >{};
                    std::copy_n(std::istream_iterator< val_t >(file), 3, coords.begin());
                    node_data.emplace(gmsh_id, detail::NodeInfo{l3ster_id++, Point{coords}});
                }
            };

            skip_until_section("$EndNodes");
            return node_data;
        };

        return parse_nodes_asciiv4();
        // TODO: switch over other possible formats once implemented
    };

    const auto parse_elements = [&](Format, const entity_data_t& entity_data, const node_data_t& node_data) {
        skip_until_section("$Elements");

        auto domain_map = MeshPartition< 1 >::domain_map_t{};

        const auto parse_elements_asciiv4 = [&]() -> MeshPartition< 1 > {
            size_t n_blocks, n_elements, min_element_tag, max_element_tag;
            file >> n_blocks >> n_elements >> min_element_tag >> max_element_tag;

            el_id_t element_id = 0;
            for (size_t block = 0; block < n_blocks; ++block)
            {
                int    entity_dim, entity_tag, element_type;
                size_t block_size;
                file >> entity_dim >> entity_tag >> element_type >> block_size;

                auto& block_domain = domain_map[entity_data.first[entity_dim].at(entity_tag)];

                const auto push_elements = [&]< size_t I >(std::integral_constant< size_t, I >) {
                    constexpr auto el_type = detail::lookupElt< I >();
                    constexpr auto el_dim  = ElementTraits< Element< el_type, 1 > >::native_dim;
                    if (block_domain.dim == Domain< 1 >::uninitialized_dim)
                        block_domain.dim = el_dim;
                    util::throwingAssert(block_domain.dim == el_dim,
                                         "Domain contains elements of different dimensions");

                    auto&      el_vec        = block_domain.elements.getVector< Element< el_type, 1 > >();
                    const auto back_inserter = std::back_inserter(el_vec);
                    const auto read_element  = [&] {
                        size_t element_tag;
                        file >> element_tag; // discard tag

                        auto nodes_gmsh = std::array< size_t, Element< el_type, 1 >::n_nodes >{};
                        std::copy_n(std::istream_iterator< size_t >(file), nodes_gmsh.size(), nodes_gmsh.begin());

                        auto node_info = std::array< detail::NodeInfo, Element< el_type, 1 >::n_nodes >{};
                        std::ranges::transform(nodes_gmsh, node_info.begin(), [&](auto n) { return node_data.at(n); });

                        return detail::makeElementFromNodeData< el_type >(node_info, element_id++);
                    };
                    std::generate_n(back_inserter, block_size, read_element);
                };

                switch (element_type)
                {
                case 1:
                    push_elements(std::integral_constant< size_t, 1 >{});
                    break;
                case 3:
                    push_elements(std::integral_constant< size_t, 3 >{});
                    break;
                case 5:
                    push_elements(std::integral_constant< size_t, 5 >{});
                    break;
                default:
                    util::throwingAssert(false, "Error while reading .msh file: Encountered unsupported element type");
                }
            }
            skip_until_section("$EndElements");

            // Sort elements by ID
            for (auto& domain : domain_map | std::views::values)
                domain.elements.visitVectors(
                    [](auto& vec) { std::ranges::sort(vec, {}, [](const auto& el) { return el.getId(); }); });

            return {std::move(domain_map), boundary_ids};
        };

        return parse_elements_asciiv4();
        // TODO: switch over other possible formats once implemented
    };

    assert_file_ok("Failed to open .msh file");
    const auto format_data = parse_format();
    const auto entity_data = parse_entities(format_data);
    const auto node_data   = parse_nodes(format_data);
    return parse_elements(format_data, entity_data, node_data);
}
} // namespace lstr::mesh
#endif // L3STER_MESH_READMESH_HPP
