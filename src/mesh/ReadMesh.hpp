#ifndef L3STER_INCGUARD_MESH_READMESH_HPP
#define L3STER_INCGUARD_MESH_READMESH_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <iostream>

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

template < MeshFormat FMT >
std::shared_ptr< Mesh > readMesh(const char* file_path, MeshFormatTag< FMT >);

namespace helpers
{
template < ElementTypes ELTYPE >
Element< ELTYPE, 1 > parse_element(std::ifstream& f)
{
    constexpr auto n_nodes = ElementTraits< Element< ELTYPE, 1 > >::nodes_per_element;
    std::array< size_t, n_nodes > nodes{};
    for (auto& node : nodes)
        f >> node;
    return Element< ELTYPE, 1 >{nodes};
}
} // namespace helpers

template <>
std::shared_ptr< Mesh > readMesh(const char* file_path, MeshFormatTag< MeshFormat::Gmsh >)
{
    // Define parsing lambdas
    const auto throw_error = [&file_path](const char* message) {
        std::stringstream error_msg;
        error_msg << "Error: " << message << "\nWhile trying to read: " << file_path;
        throw std::invalid_argument{error_msg.str()};
    };

    std::ifstream file;

    const auto skip_until_section = [&](const std::string& section_name,
                                        const char*        err_msg = "Invalid gmsh mesh file") {
        constexpr size_t line_buffer_size = 128;
        char             line_buffer[line_buffer_size];
        std::fill(line_buffer, line_buffer + sizeof line_buffer, '0');

        const auto is_buffer_equal = [&line_buffer](const std::string& str) {
            return std::equal(str.cbegin(), str.cend(), line_buffer);
        };

        const auto skip_until_char = [&file](char character) {
            constexpr auto max_line_width = std::numeric_limits< std::streamsize >::max();
            char           test_char      = static_cast< char >(file.peek());
            while (test_char != character)
            {
                file.ignore(max_line_width, '\n');
                test_char = static_cast< char >(file.peek());
            }
        };

        while (!is_buffer_equal(section_name))
        {
            if (!file.good())
                throw_error(err_msg);
            skip_until_char('$');
            file.getline(line_buffer, sizeof line_buffer);
        }
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

        Format f = [&]() -> Format {
            Format ret_value = Format::BIN32_V2;

            if (version >= 4.0 && version < 5.0)
            {
                if (bin)
                {
                    switch (size)
                    {
                    case 4:
                        ret_value = Format::BIN32_V4;
                        break;
                    case 8:
                        ret_value = Format::BIN64_V4;
                        break;
                    default:
                        throw_error("Unsupported size of size_t in the format section");
                    }
                }
                else
                    ret_value = Format::ASCII_V4;
            }
            else
            {
                if (version >= 2.0 && version < 3.0)
                {
                    if (bin)
                    {
                        switch (size)
                        {
                        case 4:
                            ret_value = Format::BIN32_V2;
                            break;
                        case 8:
                            ret_value = Format::BIN64_V2;
                            break;
                        default:
                            throw_error("Unsupported size of size_t in the format section");
                        }
                    }
                    else
                        ret_value = Format::ASCII_V2;
                }
                else
                    throw_error("Unsupported .msh format version");
            }

            return ret_value;
        }();

        if (f != Format::ASCII_V4)
            throw_error("Only the ASCII v4 gmsh format is currently supported");

        return f;
    };
    using format_data_t = Format;

    const auto parse_entities = [&](const format_data_t& format_data) {
        skip_until_section("$Entities");

        constexpr size_t                     n_entity_types = 4;
        std::array< size_t, n_entity_types > n_entities{};
        using entity_data_t = std::array< std::map< int, types::d_id_t >, n_entity_types >;
        entity_data_t entity_data{};

        const auto parse_entities_asciiv4 = [&]() {
            for (auto& e : n_entities)
                file >> e;

            const auto parse_dim_entities = [&](const size_t dim) {
                for (size_t entity = 0; entity < n_entities[dim]; entity++)
                {
                    int    entity_tag, physical_tag, skip;
                    double coord;
                    size_t n_physical_tags, n_bounding_entities;

                    file >> entity_tag >> coord >> coord >> coord;
                    if (dim > 0)
                        file >> coord >> coord >> coord;

                    file >> n_physical_tags;
                    if (n_physical_tags > 1)
                    {
                        std::stringstream error_msg;
                        error_msg << "Entity of dimension " << dim << " and tag " << entity_tag
                                  << " has more than one physical tag";
                        throw_error(error_msg.str().c_str());
                    }
                    else if (n_physical_tags == 1)
                    {
                        file >> physical_tag;
                        entity_data[dim][entity_tag] = static_cast< types::d_id_t >(physical_tag);
                    }

                    if (dim > 0)
                    {
                        file >> n_bounding_entities;
                        for (size_t i = 0; i < n_bounding_entities; ++i)
                            file >> skip;
                    }
                }
            };

            for (size_t dim = 0; dim < n_entity_types; ++dim)
                parse_dim_entities(dim);

            skip_until_section("$EndEntities");
        };

        switch (format_data)
        {
        case Format::ASCII_V4:
            parse_entities_asciiv4();
            break;
        case Format::ASCII_V2:
        case Format::BIN32_V2:
        case Format::BIN64_V2:
        case Format::BIN32_V4:
        case Format::BIN64_V4:
        default:
            throw_error("Only the ASCII v4 gmsh format is currently supported");
        }

        const size_t n_physical_domains = [&]() {
            const size_t n_physical_entities = std::transform_reduce(
                entity_data.cbegin(), entity_data.cend(), 0, std::plus<>{}, [](const auto& map) {
                    return map.size();
                });
            std::vector< types::d_id_t > unique_physical_ids;
            unique_physical_ids.reserve(n_physical_entities);
            auto upi_inserter = std::back_inserter(unique_physical_ids);
            std::for_each(
                entity_data.cbegin(), entity_data.cend(), [&upi_inserter](const auto& map) {
                    std::transform(map.cbegin(),
                                   map.cend(),
                                   upi_inserter,
                                   [](const auto& map_entry) { return map_entry.second; });
                });
            std::sort(unique_physical_ids.begin(), unique_physical_ids.end());
            unique_physical_ids.erase(
                std::unique(unique_physical_ids.begin(), unique_physical_ids.end()),
                unique_physical_ids.end());
            return unique_physical_ids.size();
        }();

        return std::make_pair(entity_data, n_physical_domains);
    };
    using entity_data_t = std::invoke_result_t< decltype(parse_entities), const format_data_t& >;

    using node_data_t      = std::pair< bool, std::vector< std::pair< size_t, Node< 3 > > > >;
    const auto parse_nodes = [&](const format_data_t& format_data) -> node_data_t {
        skip_until_section("$Nodes", "'Node' section not found");

        node_data_t node_data{};

        const auto parse_nodes_asciiv4 = [&]() {
            size_t n_blocks, n_nodes, min_node_tag, max_node_tag;
            file >> n_blocks >> n_nodes >> min_node_tag >> max_node_tag;

            node_data.first = max_node_tag - min_node_tag == n_nodes - 1;
            node_data.second.resize(n_nodes);
            auto node_it = node_data.second.begin();

            const auto parse_block = [&]() {
                int    dim, entity_tag;
                bool   parametric;
                size_t block_size;
                file >> dim >> entity_tag >> parametric >> block_size;

                if (parametric)
                    throw_error("Parametric nodes are unsupported");

                const auto block_begin = node_it;
                for (size_t tag_ind = 0; tag_ind < block_size; ++tag_ind)
                    file >> node_it++->first;

                node_it = block_begin;
                for (size_t tag_ind = 0; tag_ind < block_size; ++tag_ind)
                {
                    for (auto& coord : node_it++->second.coords)
                        file >> coord;
                }
            };

            for (size_t block_ind = 0; block_ind < n_blocks; ++block_ind)
                parse_block();

            skip_until_section("$EndNodes");
        };

        switch (format_data)
        {
        case Format::ASCII_V4:
            parse_nodes_asciiv4();
            break;
        case Format::ASCII_V2:
        case Format::BIN32_V2:
        case Format::BIN64_V2:
        case Format::BIN32_V4:
        case Format::BIN64_V4:
        default:
            throw_error("Only the ASCII v4 gmsh format is currently supported");
        }
        return node_data;
    };

    const auto parse_elements = [&](const format_data_t& format_data,
                                    const entity_data_t& entity_data) -> MeshPartition {
        skip_until_section("$Elements", "'Elements' section not found");

        typename MeshPartition::domain_map_t domain_map;

        const auto parse_elements_asciiv4 = [&]() {
            size_t n_blocks, n_elements, min_element_tag, max_element_tag;
            file >> n_blocks >> n_elements >> min_element_tag >> max_element_tag;

            const auto parse_block = [&]() {
                int    entity_dim, entity_tag, element_type;
                size_t block_size;
                file >> entity_dim >> entity_tag >> element_type >> block_size;

                const types::d_id_t block_physical_id =
                    entity_data.first[entity_dim].at(entity_tag);
                auto& block_domain = domain_map[block_physical_id];

                for (size_t element = 0; element < block_size; ++element)
                {
                    size_t element_tag;
                    file >> element_tag;

                    switch (element_type)
                    {
                    case 1:
                        block_domain.pushBack(helpers::parse_element< ElementTypes::Line >(file));
                        break;
                    case 3:
                        block_domain.pushBack(helpers::parse_element< ElementTypes::Quad >(file));
                        break;
                    default:
                        std::stringstream err;
                        err << "Unsupported element type: " << element_type;
                        throw_error(err.str().c_str());
                    }
                }
            };

            for (size_t block = 0; block < n_blocks; ++block)
                parse_block();

            return domain_map;
        };

        switch (format_data)
        {
        case Format::ASCII_V4:
            parse_elements_asciiv4();
            break;
        case Format::ASCII_V2:
        case Format::BIN32_V2:
        case Format::BIN64_V2:
        case Format::BIN32_V4:
        case Format::BIN64_V4:
        default:
            throw_error("Only the ASCII v4 gmsh format is currently supported");
        }

        skip_until_section("$EndElements");
        return MeshPartition{std::move(domain_map)};
    };

    const auto make_contiguous_mesh = [&](node_data_t& node_data, MeshPartition& part) {
        auto& [is_contiguous, node_vector] = node_data;
        std::sort(node_vector.begin(), node_vector.end(), [](const auto& p1, const auto& p2) {
            return p1.first < p2.first;
        });
        std::vector< Node< 3 > > nodes;
        if (is_contiguous)
        {
            auto node_inserter = std::back_inserter(nodes);
            std::transform(node_vector.cbegin(),
                           node_vector.cend(),
                           node_inserter,
                           [](const auto& node_data_entry) { return node_data_entry.second; });

            part.visitAllElements([min_node_tag = node_vector.front().first](auto& element) {
                std::for_each(element.getNodesRef().begin(),
                              element.getNodesRef().end(),
                              [&min_node_tag](auto& node) { node -= min_node_tag; });
            });
        }
        else
            throw_error("Node numbering must start at 0 and be contiguous [TO DO]");

        return Mesh{std::move(nodes), std::move(part)};
    };

    // Parse
    file.open(file_path);
    if (!file.is_open())
        throw_error("Could not open gmsh mesh file");

    const auto format_data  = parse_format();
    const auto entity_data  = parse_entities(format_data);
    auto       node_data    = parse_nodes(format_data);
    auto       element_data = parse_elements(format_data, entity_data);

    return std::make_shared< Mesh >(make_contiguous_mesh(node_data, element_data));
} // namespace lstr::mesh
} // namespace lstr::mesh

#endif // L3STER_INCGUARD_MESH_READMESH_HPP
