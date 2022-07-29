#ifndef L3STER_POST_VTKEXPORT_HPP
#define L3STER_POST_VTKEXPORT_HPP

#include "l3ster/assembly/NodeToDofMap.hpp"
#include "l3ster/util/StringUtils.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

#include <string>
#include <string_view>

namespace lstr
{
namespace detail::vtk
{
template < ElementTypes ET, el_o_t EO >
consteval size_t numSerialEntriesTopo()
{
    constexpr auto el_o = static_cast< size_t >(EO);
    if constexpr (ET == ElementTypes::Line)
        return el_o;
    else if constexpr (ET == ElementTypes::Quad)
        return el_o * el_o;
    else if constexpr (ET == ElementTypes::Hex)
        return el_o * el_o * el_o;
}
template < ElementTypes ET, el_o_t EO >
consteval size_t numSerialIntegersTopo()
{
    if constexpr (ET == ElementTypes::Line)
        return 3 * numSerialEntriesTopo< ET, EO >();
    else if constexpr (ET == ElementTypes::Quad)
        return 5 * numSerialEntriesTopo< ET, EO >();
    else if constexpr (ET == ElementTypes::Hex)
        return 9 * numSerialEntriesTopo< ET, EO >();
}

template < detail::ProblemDef_c auto problem_def >
std::array< size_t, 2 > getLocalTopoSize(const MeshPartition& mesh, ConstexprValue< problem_def >)
{
    constexpr auto get_el_entries = []< ElementTypes ET, el_o_t EO >(const Element< ET, EO >&) {
        constexpr auto retval = std::make_pair(numSerialEntriesTopo< ET, EO >(), numSerialIntegersTopo< ET, EO >());
        return retval;
    };
    return mesh.reduce(
        std::array< size_t, 2 >{0, 0},
        get_el_entries,
        [](const std::array< size_t, 2 > a1, std::array< size_t, 2 > a2) {
            return std::array< size_t, 2 >{a1[0] + a2[0], a1[1] + a2[1]};
        },
        std::views::keys(problem_def));
}

auto getCellOffsetsRequest(const MpiComm& comm, const std::array< size_t, 2 >& local_topo_sizes)
{
    auto offsets = std::make_unique_for_overwrite< size_t[] >(4); // 2 for recvbuf, 2 for sendbuf; NOLINT
}

template < el_o_t EO, std::output_iterator< int > Iter >
auto serializeSubtopology(const Element< ElementTypes::Line, EO >& element, Iter out)
{}
template < el_o_t EO, std::output_iterator< int > Iter >
auto serializeSubtopology(const Element< ElementTypes::Quad, EO >& element, Iter out)
{}
template < el_o_t EO, std::output_iterator< int > Iter >
auto serializeSubtopology(const Element< ElementTypes::Hex, EO >& element, Iter out)
{}

inline constexpr size_t max_num_chars = 20; // used to make section header sizes independent of contents

inline std::string makeFileHeader(std::string_view name = std::string_view{"L3STER results"})
{
    using namespace std::string_view_literals;
    constexpr size_t max_name_len = 256;
    constexpr auto   line1        = "# vtk DataFile Version 2.0\n"sv;
    constexpr auto   line3        = "BINARY\nDATASET UNSTRUCTURED_GRID\n"sv;
    std::string      retval;
    retval.reserve(line1.size() + std::min< size_t >(name.size(), max_name_len) + line3.size());
    retval.append(line1);
    retval.append(name.substr(0, max_name_len));
    retval.append(line3);
    return retval;
}
inline std::string makePointsHeader(size_t n_points)
{
    std::string retval{"\nPOINTS "};
    retval.append(prependSpaces(std::to_string(n_points), max_num_chars));
    retval.append(" ");
    retval.append("\n");
    return retval;
}
inline std::string makeCellsHeader(size_t n_cells, size_t cell_section_size)
{
    std::string retval{"\nCELLS "};
    retval.append(prependSpaces(std::to_string(n_cells), max_num_chars));
    retval.append(" ");
    retval.append(prependSpaces(std::to_string(cell_section_size), max_num_chars));
    retval.append("\n");
    return retval;
}
inline std::string makeCellTypesHeader(size_t n_cells)
{
    std::string retval{"\nCELL_TYPES "};
    retval.append(prependSpaces(std::to_string(n_cells), max_num_chars));
    retval.append("\n");
    return retval;
}
} // namespace detail::vtk

void exportResultsToVtk()
{}
} // namespace lstr
#endif // L3STER_POST_VTKEXPORT_HPP
