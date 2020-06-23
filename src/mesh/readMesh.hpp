#ifndef L3STER_INCGUARD_MESH_READMESH_HPP
#define L3STER_INCGUARD_MESH_READMESH_HPP

#include "mesh/MeshPartition.hpp"
#include "mesh/Node.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
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
        char           test_char      = file.peek();
        while (test_char != character)
        {
            file.ignore(max_line_width, '\n');
            test_char = file.peek();
        }
    };

    constexpr size_t line_buffer_size = 256;
    char             line_buffer[line_buffer_size];
    std::fill(line_buffer, line_buffer + line_buffer_size, '0');

    const auto is_buffer_equal = [&line_buffer](const std::string& str) {
        return std::equal(str.cbegin(), str.cend(), line_buffer);
    };

    const auto skip_until_section = [&](const std::string& section_name,
                                        const char*        err_msg = "Invalid gmsh mesh file") {
        while (!is_buffer_equal(section_name))
        {
            if (!file.good())
                throw_error(err_msg);
            skip_until_char('$');
            file.getline(line_buffer, sizeof line_buffer);
        }
    };

    skip_until_section("$MeshFormat");

    constexpr float min_readable_version = 4.;
    float           format_version;
    file >> format_version;

    if (format_version < min_readable_version)
        throw_error("Only gmsh format version 4.0 or newer is supported");

    bool binary_format_flag;
    file >> binary_format_flag;
    if (binary_format_flag)
        throw_error("Only ASCII gmsh format is supported");

    // TO DO: IMPLEMENT BINARY FORMAT SUPPORT

    int data_size;
    file >> data_size;
    if (binary_format_flag && data_size != 8)
        throw_error("Only data size 8 (64 bit) is supported for the binary format");

    skip_until_section("$EndMeshFormat");
    skip_until_section("$Entities");

    size_t n_points, n_curves, n_surfs, n_vols;
    file >> n_points >> n_curves >> n_surfs >> n_vols;

    double z_min = std::numeric_limits< double >::max(),
           z_max = std::numeric_limits< double >::min();

    // Parse points
    for (size_t i = 0; i < n_points; ++i)
    {
        int    read_int;
        size_t read_size_t;
        double read_double;

        file >> read_int;

        file >> read_double >> read_double >> read_double;
        z_min = std::min(z_min, read_double);
        z_max = std::max(z_max, read_double);

        file >> read_size_t >> read_int;
    }

    // Parse curves

    Mesh ret_mesh;
    return ret_mesh;
}
} // namespace lstr::mesh

#endif // L3STER_INCGUARD_MESH_READMESH_HPP
