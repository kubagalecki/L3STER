#ifndef L3STER_MESH_READMESH_HPP
#define L3STER_MESH_READMESH_HPP

#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/Meta.hpp"

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

namespace lstr
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
    std::pair{1u, ElementTypes::Line}, std::pair{3u, ElementTypes::Quad}, std::pair{5u, ElementTypes::Hex}};

template < size_t I >
consteval ElementTypes lookupElt()
{
    return std::ranges::find(gmsh_elt_lookup_tab, I, [](const auto p) { return p.first; })->second;
}

template < ElementTypes T >
void reorderNodes(auto& nodes)
{
    if constexpr (T == ElementTypes::Quad)
        std::swap(nodes[2], nodes[3]);
    else if constexpr (T == ElementTypes::Hex)
    {
        std::swap(nodes[2], nodes[3]);
        std::swap(nodes[6], nodes[7]);
    }
}
} // namespace detail

inline auto readMesh(std::string_view file_path, MeshFormatTag< MeshFormat::Gmsh >) -> MeshPartition< 1 >
{
    L3STER_PROFILE_FUNCTION;
    const auto throw_error = [&file_path](std::string_view     message,
                                          std::source_location src_loc = std::source_location::current()) {
        std::stringstream error_msg;
        error_msg << "Error: " << message << "\nWhile trying to read: " << file_path;
        util::throwingAssert(false, error_msg.view(), src_loc);
    };

    std::ifstream file;

    const auto skip_until_section = [&](std::string_view section_name,
                                        std::string_view err_msg = "Invalid gmsh mesh file") {
        std::search(std::istream_iterator< char >(file),
                    std::istream_iterator< char >{},
                    section_name.cbegin(),
                    section_name.cend());
        util::throwingAssert(file.good(), err_msg);
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
        skip_until_section("$MeshFormat", "'MeshFormat' section not found");

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
                    throw_error("Unsupported size of size_t in the format section");
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
                        throw_error("Unsupported size of size_t in the format section");
                    }
                }
                else
                    format = Format::ASCII_V2;
            }
            else
                throw_error("Unsupported .msh format version");
        }

        if (format != Format::ASCII_V4)
            throw_error("Only the ASCII v4 gmsh format is currently supported");

        return format;
    };

    const auto parse_entities = [&](const Format&) {
        skip_until_section("$Entities");

        constexpr size_t                     n_entity_types = 4;
        std::array< size_t, n_entity_types > n_entities{};
        using entity_data_t = std::array< std::map< int, d_id_t >, n_entity_types >;
        entity_data_t entity_data{};

        const auto parse_entities_asciiv4 = [&]() {
            for (auto& e : n_entities)
                file >> e;

            const auto parse_dim_entities = [&](const size_t dim) {
                const auto parse_entity = [&]() {
                    int    entity_tag;
                    double coord;
                    file >> entity_tag;
                    for (size_t n_io = 0; n_io < 3 + ((dim > 0) * 3); ++n_io)
                        file >> coord;

                    size_t n_physical_tags;
                    file >> n_physical_tags;
                    if (n_physical_tags > 1)
                    {
                        std::stringstream error_msg;
                        error_msg << "Entity of dimension " << dim << " and tag " << entity_tag
                                  << " has more than one physical tag";
                        throw_error(error_msg.view());
                    }
                    else if (n_physical_tags == 1)
                    {
                        int physical_tag;
                        file >> physical_tag;
                        entity_data[dim][entity_tag] = static_cast< d_id_t >(physical_tag);
                    }

                    if (dim > 0)
                    {
                        size_t n_bounding_entities;
                        file >> n_bounding_entities;
                        int skip;
                        for (size_t i = 0; i < n_bounding_entities; ++i)
                            file >> skip;
                    }
                };

                for (size_t entity = 0; entity < n_entities[dim]; entity++)
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

    using node_data_t      = std::tuple< std::vector< size_t >, std::vector< Point< 3 > >, bool >;
    const auto parse_nodes = [&](const Format&) -> node_data_t {
        skip_until_section("$Nodes", "'Node' section not found");

        const auto parse_nodes_asciiv4 = [&]() {
            size_t n_blocks, n_nodes, min_node_tag, max_node_tag;
            file >> n_blocks >> n_nodes >> min_node_tag >> max_node_tag;

            node_data_t node_data{};
            auto& [node_ids, node_coords, is_contiguous] = node_data;

            is_contiguous = max_node_tag - min_node_tag == n_nodes - 1;
            node_ids.reserve(n_nodes);
            node_coords.reserve(n_nodes);

            for (size_t block_ind = 0; block_ind < n_blocks; ++block_ind)
            {
                std::tuple< size_t, int, int, bool > _{};
                auto& [block_size, dim, entity_tag, parametric] = _;
                file >> dim >> entity_tag >> parametric >> block_size;

                if (parametric)
                    throw_error("Parametric nodes are unsupported");

                std::copy_n(std::istream_iterator< size_t >(file), block_size, std::back_inserter(node_ids));
                std::generate_n(std::back_inserter(node_coords), block_size, [&] {
                    std::array< val_t, 3 > retval; // NOLINT unneeded initialization
                    std::copy_n(std::istream_iterator< val_t >(file), 3, retval.begin());
                    return Point< 3 >{retval};
                });
            };

            const auto srt_ind = util::sortingPermutation(node_ids.cbegin(), node_ids.cend());
            {
                auto temp = std::vector< size_t >(n_nodes);
                util::copyPermuted(node_ids.cbegin(), node_ids.cend(), srt_ind.cbegin(), begin(temp));
                node_ids = std::move(temp);
            }
            {
                auto temp = std::vector< Point< 3 > >(n_nodes);
                util::copyPermuted(node_coords.cbegin(), node_coords.cend(), srt_ind.cbegin(), begin(temp));
                node_coords = std::move(temp);
            }

            skip_until_section("$EndNodes");
            return node_data;
        };

        return parse_nodes_asciiv4();
        // TODO: switch over other possible formats once implemented
    };

    const auto parse_elements = [&](Format, const entity_data_t& entity_data, const node_data_t& node_data) {
        skip_until_section("$Elements", "'Elements' section not found");

        auto domain_map = MeshPartition< 1 >::domain_map_t{};

        const auto parse_elements_asciiv4 = [&]() -> MeshPartition< 1 > {
            size_t n_blocks, n_elements, min_element_tag, max_element_tag, element_id = 0;
            file >> n_blocks >> n_elements >> min_element_tag >> max_element_tag;

            const auto node_contig_index = [&](n_id_t gmsh_id) -> n_id_t {
                const auto& [ids, coords, is_contig] = node_data;
                if (is_contig)
                    return gmsh_id - ids.front();
                else
                    return std::distance(begin(ids), std::ranges::find(ids, gmsh_id));
            };
            const auto lookup_node_coords = [&](n_id_t gmsh_id) {
                const auto& [ids, coords, is_contig] = node_data;
                return coords[node_contig_index(gmsh_id)];
            };

            for (size_t block = 0; block < n_blocks; ++block)
            {
                int    entity_dim, entity_tag, element_type;
                size_t block_size;
                file >> entity_dim >> entity_tag >> element_type >> block_size;

                auto& block_domain = domain_map[entity_data.first[entity_dim].at(entity_tag)];

                const auto push_elements = [&]< size_t I >(std::integral_constant< size_t, I >) {
                    constexpr auto el_type = detail::lookupElt< I >();
                    std::generate_n(block_domain.getBackInserter< el_type, 1 >(), block_size, [&] {
                        size_t element_tag;
                        file >> element_tag; // discard tag
                        std::array< n_id_t, Element< el_type, 1 >::n_nodes > nodes;
                        std::copy_n(std::istream_iterator< n_id_t >(file), nodes.size(), begin(nodes));
                        detail::reorderNodes< el_type >(nodes);
                        typename ElementData< el_type, 1 >::ElementData::vertex_array_t data;
                        std::ranges::transform(nodes, begin(data), lookup_node_coords);
                        std::ranges::for_each(nodes, [&](auto& n) { n = node_contig_index(n); });
                        return Element< el_type, 1 >{nodes, ElementData< el_type, 1 >{data}, element_id++};
                    });
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
                    std::stringstream err;
                    err << "Unsupported element type: " << element_type;
                    throw_error(err.view());
                }
            }
            skip_until_section("$EndElements");
            return {std::move(domain_map)};
        };

        return parse_elements_asciiv4();
        // TODO: switch over other possible formats once implemented
    };

    // Parse
    file.open(std::filesystem::path{file_path});
    util::throwingAssert(file.is_open(), "Could not open mesh file");

    const auto format_data = parse_format();
    const auto entity_data = parse_entities(format_data);
    const auto node_data   = parse_nodes(format_data);
    return parse_elements(format_data, entity_data, node_data);
}
} // namespace lstr
#endif // L3STER_MESH_READMESH_HPP
