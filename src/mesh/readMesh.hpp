#ifndef L3STER_INCGUARD_MESH_READMESH_HPP
#define L3STER_INCGUARD_MESH_READMESH_HPP

#include "mesh/Mesh.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <tuple>
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
Mesh readMesh(const std::filesystem::path& file_path, MeshFormatTag< FMT >);

template <>
Mesh readMesh(const std::filesystem::path& file_path, MeshFormatTag< MeshFormat::Gmsh >)
{
    const auto throw_error = [&file_path](const char* message) {
        std::stringstream error_msg;
        error_msg << message << ":\n" << file_path;
        throw std::invalid_argument{error_msg.str()};
    };

    std::ifstream file{file_path};
    if (!file.is_open())
        throw_error("Could not open gmsh mesh file");

    const auto skip_until_char = [&file](char character) {
        constexpr auto max_line_width = std::numeric_limits< std::streamsize >::max();
        char           test_char      = static_cast< char >(file.peek());
        while (test_char != character)
        {
            file.ignore(max_line_width, '\n');
            test_char = static_cast< char >(file.peek());
        }
    };

    const auto skip_until_section = [&](const std::string& section_name,
                                        const char*        err_msg = "Invalid gmsh mesh file") {
        constexpr size_t line_buffer_size = 128;
        char             line_buffer[line_buffer_size];
        std::fill(line_buffer, line_buffer + sizeof line_buffer, '0');

        const auto is_buffer_equal = [&line_buffer](const std::string& str) {
            return std::equal(str.cbegin(), str.cend(), line_buffer);
        };

        while (!is_buffer_equal(section_name))
        {
            if (!file.good())
                throw_error(err_msg);
            skip_until_char('$');
            file.getline(line_buffer, sizeof line_buffer);
        }
    };

    const auto parse_format = [&]() {
        skip_until_section("$MeshFormat");

        std::tuple< float, bool, unsigned int > format_data;
        auto& [version, bin, size] = format_data;
        file >> version >> bin >> size;

        skip_until_section("$EndMeshFormat");

        return format_data;
    };

    const auto parse_entities_ascii4 = [&]() {
        skip_until_section("$Entities");

        constexpr size_t                                             n_entity_types = 4;
        std::array< size_t, n_entity_types >                         n_entities;
        std::array< std::map< int, types::d_id_t >, n_entity_types > entity_data;

        double z_min = std::numeric_limits< double >::max();
        double z_max = std::numeric_limits< double >::min();

        const auto parse_dim_entities = [&](const size_t dim) {
            for (size_t entity = 0; entity < n_entities[dim]; entity++)
            {
                int    entity_tag, physical_tag, skip;
                double coord;
                size_t n_physical_tags, n_bounding_entities;

                file >> entity_tag >> coord >> coord >> coord;
                z_min = std::min(z_min, coord);
                if (dim > 0)
                    file >> coord >> coord >> coord;
                z_max = std::max(z_max, coord);

                file >> n_physical_tags;
                if (n_physical_tags > 1)
                {
                    std::stringstream error_msg;
                    error_msg << "Entity of dimension " << dim << "and tag " << entity_tag
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

        size_t dim_counter = 0; // use of C++20 ranged for init possible
        for (const auto& it_n_entities : n_entities)
            parse_dim_entities(dim_counter++);

        constexpr double eps_tolerance = 10.;
        const bool is_3d = z_max - z_min > eps_tolerance * std::numeric_limits< double >::epsilon();

        skip_until_section("$EndEntities");

        return entity_data;
    };

    const auto format_data = parse_format();

    const auto entity_data = [&]() {
        if (std::get< 1 >(format_data))
            throw_error("Binary gmsh format is not supported");
        // TO DO: IMPLEMENT IMPORT FROM BINARY
        else if (std::get< 0 >(format_data) < 4.0 || std::get< 0 >(format_data) >= 5.0)
            throw_error("Only gmsh format 4 is supported");
        // TO DO: IMPLEMENT GMSH V2 SUPPORT
        return parse_entities_ascii4();
    }();

    Mesh ret_mesh;
    return ret_mesh;
} // namespace lstr::mesh
} // namespace lstr::mesh

#endif // L3STER_INCGUARD_MESH_READMESH_HPP
