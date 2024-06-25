#ifndef L3STER_POST_VTKEXPORT_HPP
#define L3STER_POST_VTKEXPORT_HPP

#include "l3ster/dofs/NodeToDofMap.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"
#include "l3ster/post/SolutionManager.hpp"
#include "l3ster/util/Base64.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace lstr
{
class PvtuExporter
{
public:
    template < el_o_t... orders >
    explicit PvtuExporter(const mesh::MeshPartition< orders... >& mesh);
    template < std::ranges::range FieldCompInds >
    void exportSolution(std::string_view                       file_name,
                        const MpiComm&                         comm,
                        const SolutionManager&                 solution_manager,
                        const util::ArrayOwner< std::string >& field_names,
                        FieldCompInds&&                        field_component_inds)
        requires RangeOfConvertibleTo_c< std::ranges::range_reference_t< FieldCompInds >, size_t >;
    template < el_o_t... orders >
    void        updateNodeCoords(const mesh::MeshPartition< orders... >& mesh);
    inline void flushWriteQueue();

private:
    struct SectionSizes
    {
        size_t topology, offsets, cell_types;
    };

    using payload_t = std::variant< std::monostate, std::string, util::ArrayOwner< char > >;
    struct AsyncWrite
    {
        // Member declaration order ensures correct behavior on destruction: 1) block 2) dealloc payload 3) close file
        std::shared_ptr< MpiComm::FileHandle > file_handle;
        payload_t                              payload;
        MpiComm::Request                       request;
    };

    template < el_o_t... orders >
    void initTopo(const mesh::MeshPartition< orders... >& mesh);

    inline void enqueuePvtuFileWrite(std::string_view                                      file_name,
                                     const MpiComm&                                        comm,
                                     const util::ArrayOwner< std::string >&                field_names,
                                     const util::ArrayOwner< util::ArrayOwner< size_t > >& field_component_inds);
    inline void enqueueVtuFileWrite(std::string_view                                      file_name,
                                    const MpiComm&                                        comm,
                                    const SolutionManager&                                solution_manager,
                                    const util::ArrayOwner< std::string >&                field_names,
                                    const util::ArrayOwner< util::ArrayOwner< size_t > >& field_component_inds);
    inline auto makeDataDescription(const util::ArrayOwner< std::string >&                field_names,
                                    const util::ArrayOwner< util::ArrayOwner< size_t > >& field_component_inds,
                                    const std::vector< util::ArrayOwner< char > >& encoded_fields) const -> std::string;

    template < ContiguousSizedRangeOf< char > Text >
    MPI_Offset enqueueWrite(std::shared_ptr< MpiComm::FileHandle > file, MPI_Offset pos, Text&& text)
        requires std::ranges::borrowed_range< Text > or std::same_as< std::string, Text > or
                 std::same_as< util::ArrayOwner< char >, Text >;

    size_t                    m_n_cells, m_n_nodes;
    SectionSizes              m_section_sizes;
    std::string               m_encoded_topo, m_encoded_coords;
    std::vector< AsyncWrite > m_write_queue; // this needs to be destroyed before the encoded data
};

namespace post::vtk
{
// The .pvtu top level file
inline constexpr std::string_view pvtu_preamble  = R"(<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="1.0" byte_order="LittleEndian">
<PUnstructuredGrid GhostLevel="0">
<PPoints>
  <PDataArray type="Float64" Name="Position" NumberOfComponents="3"/>
</PPoints>
<PCells>
  <PDataArray type="UInt32" Name="connectivity" NumberOfComponents="1"/>
  <PDataArray type="UInt32" Name="offsets" NumberOfComponents="1"/>
  <PDataArray type="UInt8" Name="types" NumberOfComponents="1"/>
</PCells>
<PPointData)";
inline constexpr std::string_view pvtu_postamble = R"(</PUnstructuredGrid>
</VTKFile>)";

inline void appendPvtuPointDataDef(std::string&                           str,
                                   const util::ArrayOwner< std::string >& field_names,
                                   const util::ArrayOwner< size_t >&      n_field_components)
{
    bool scal_present = false, vec_present = false; // Only 1 scalar and 1 vector can be present in the header
    auto name_it = std::ranges::begin(field_names);
    for (auto n_comps : n_field_components)
    {
        const bool is_vector = n_comps > 1;
        if (is_vector and not std::exchange(vec_present, true))
        {
            str += " Vectors=\"";
            str += *name_it;
            str += '"';
        }
        else if (not scal_present)
        {
            str += " Scalars=\"";
            str += *name_it;
            str += '"';
            scal_present = true;
        }
        if (scal_present and vec_present)
            break;
        ++name_it;
    }
    str += ">\n";
    name_it = std::ranges::begin(field_names);
    for (auto n_comps : n_field_components)
    {
        str += R"(  <PDataArray type="Float64" Name=")";
        str += *name_it++;
        str += R"(" NumberOfComponents=")";
        str += std::to_string(n_comps);
        str += "\"/>\n";
    }
    str += "</PPointData>\n";
    str += "<PCellData>\n";
    str += "</PCellData>\n";
}

inline void appendPvtuPieceDataDef(std::string& str, std::string_view name, int n_ranks)
{
    for (int rank = 0; rank < n_ranks; ++rank)
    {
        str += "<Piece Source=\"";
        str += std::filesystem::path{name}.stem().string();
        str += '_';
        str += std::to_string(rank);
        str += ".vtu\"/>\n";
    }
}

inline auto makePvtuFileContents(std::string_view                       name,
                                 int                                    n_ranks,
                                 const util::ArrayOwner< std::string >& field_names,
                                 const util::ArrayOwner< size_t >&      n_field_components) -> std::string
{
    std::string retval;
    retval.reserve(1u << 12);
    retval += pvtu_preamble;
    appendPvtuPointDataDef(retval, field_names, n_field_components);
    appendPvtuPieceDataDef(retval, name, n_ranks);
    retval += pvtu_postamble;
    return retval;
}

// The .vtu data file (1 per rank)
inline constexpr std::string_view vtu_preamble  = R"(<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">
<UnstructuredGrid>
)";
inline constexpr std::string_view vtu_postamble = "</AppendedData>\n</VTKFile>";

// Data serialization
template < mesh::ElementType ET, el_o_t EO >
consteval size_t numSubels()
{
    constexpr auto el_o = static_cast< size_t >(EO);
    if constexpr (ET == mesh::ElementType::Line)
        return el_o;
    else if constexpr (ET == mesh::ElementType::Quad)
        return el_o * el_o;
    else if constexpr (ET == mesh::ElementType::Hex)
        return el_o * el_o * el_o;
    else
        static_assert(ET != ET); // Assert every element type has a corresponding branch
}

template < mesh::ElementType ET, el_o_t EO >
consteval size_t numSubelNodes()
{
    if constexpr (ET == mesh::ElementType::Line)
        return 2;
    else if constexpr (ET == mesh::ElementType::Quad)
        return 4;
    else if constexpr (ET == mesh::ElementType::Hex)
        return 8;
    else
        static_assert(ET != ET); // Assert every element type has a corresponding branch
}

template < mesh::ElementType ET, el_o_t EO >
consteval size_t numSerialTopoEntries()
{
    return numSubels< ET, EO >() * numSubelNodes< ET, EO >();
}

template < mesh::ElementType ET, el_o_t EO >
consteval unsigned char subelCellType()
{
    if constexpr (ET == mesh::ElementType::Line)
        return 3;
    else if constexpr (ET == mesh::ElementType::Quad)
        return 9;
    else if constexpr (ET == mesh::ElementType::Hex)
        return 12;
    else
        static_assert(ET != ET); // Assert every element type has a corresponding branch
}

template < mesh::ElementType ET, el_o_t EO >
auto serializeElementSubtopo(const mesh::Element< ET, EO >& element)
{
    const auto&                                            nodes = element.getNodes();
    std::array< n_id_t, numSerialTopoEntries< ET, EO >() > retval;
    auto                                                   out_topo = retval.begin();
    if constexpr (ET == mesh::ElementType::Line)
    {
        for (size_t i = 0; i < nodes.size() - 1; ++i)
        {
            *out_topo++ = nodes[i];
            *out_topo++ = nodes[i + 1];
        }
    }
    else if constexpr (ET == mesh::ElementType::Quad)
    {
        constexpr auto node_per_side =
            mesh::ElementTraits< mesh::Element< mesh::ElementType::Line, EO > >::nodes_per_element;
        for (size_t row = 0; row < node_per_side - 1; ++row)
            for (size_t col = 0; col < node_per_side - 1; ++col)
            {
                const size_t base = row * node_per_side + col;
                *out_topo++       = nodes[base];
                *out_topo++       = nodes[base + 1];
                *out_topo++       = nodes[base + node_per_side + 1];
                *out_topo++       = nodes[base + node_per_side];
            }
    }
    else if constexpr (ET == mesh::ElementType::Hex)
    {
        constexpr auto nodes_per_side =
            mesh::ElementTraits< mesh::Element< mesh::ElementType::Line, EO > >::nodes_per_element;
        constexpr auto nodes_per_layer = nodes_per_side * nodes_per_side;
        for (size_t layer = 0; layer < nodes_per_side - 1; ++layer)
            for (size_t row = 0; row < nodes_per_side - 1; ++row)
                for (size_t col = 0; col < nodes_per_side - 1; ++col)
                {
                    const size_t base = layer * nodes_per_layer + row * nodes_per_side + col;
                    *out_topo++       = nodes[base];
                    *out_topo++       = nodes[base + 1];
                    *out_topo++       = nodes[base + nodes_per_side + 1];
                    *out_topo++       = nodes[base + nodes_per_side];
                    *out_topo++       = nodes[base + nodes_per_layer];
                    *out_topo++       = nodes[base + 1 + nodes_per_layer];
                    *out_topo++       = nodes[base + nodes_per_side + 1 + nodes_per_layer];
                    *out_topo++       = nodes[base + nodes_per_side + nodes_per_layer];
                }
    }
    else
        static_assert(ET != ET); // Assert every element type has a corresponding branch
    return retval;
}

template < el_o_t... orders >
std::array< size_t, 2 > getLocalTopoSize(const mesh::MeshPartition< orders... >& mesh)
{
    constexpr auto get_el_entries = []< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >&) {
        return std::array{numSubels< ET, EO >(), numSerialTopoEntries< ET, EO >()};
    };
    constexpr auto add = [](const std::array< size_t, 2 >& a1, const std::array< size_t, 2 >& a2) {
        return std::array< size_t, 2 >{a1[0] + a2[0], a1[1] + a2[1]};
    };
    constexpr auto zero = std::array< size_t, 2 >{};
    return mesh.transformReduce(mesh.getDomainIds(), zero, get_el_entries, add);
}

template < el_o_t... orders >
unsigned getLocalNodeIndex(const mesh::MeshPartition< orders... >& mesh, n_id_t node)
{
    const auto owned_find_result = std::ranges::lower_bound(mesh.getOwnedNodes(), node);
    if (owned_find_result != end(mesh.getOwnedNodes()) and *owned_find_result == node)
        return static_cast< unsigned >(std::distance(begin(mesh.getOwnedNodes()), owned_find_result));
    const auto ghost_find_result = std::ranges::lower_bound(mesh.getGhostNodes(), node);
    return static_cast< unsigned >(mesh.getOwnedNodes().size() +
                                   std::distance(begin(mesh.getGhostNodes()), ghost_find_result));
}

template < el_o_t... orders >
auto serializeTopology(const mesh::MeshPartition< orders... >& mesh)
{
    constexpr auto n_unsigned_chars = sizeof(std::uint64_t) / sizeof(unsigned char);
    constexpr auto n_unsigneds      = sizeof(std::uint64_t) / sizeof(unsigned);
    const auto [n_cells, topo_size] = getLocalTopoSize(mesh);
    std::vector< unsigned char > cell_types;
    std::vector< unsigned >      topo_data;
    std::vector< unsigned >      offsets;
    cell_types.reserve(n_cells + n_unsigned_chars);
    topo_data.reserve(topo_size + n_unsigneds);
    offsets.reserve(n_cells + n_unsigneds);

    // The first 8 bytes encode the size of the data
    std::uint64_t data_bytes = n_cells * sizeof(unsigned char);
    cell_types.resize(n_unsigned_chars);
    std::memcpy(cell_types.data(), &data_bytes, sizeof data_bytes);
    data_bytes = topo_size * sizeof(unsigned);
    topo_data.resize(n_unsigneds);
    std::memcpy(topo_data.data(), &data_bytes, sizeof data_bytes);
    data_bytes = n_cells * sizeof(unsigned);
    offsets.resize(n_unsigneds);
    std::memcpy(offsets.data(), &data_bytes, sizeof data_bytes);

    unsigned   offset          = 0;
    const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        const auto serialized_subtopo = serializeElementSubtopo(element);
        std::ranges::transform(serialized_subtopo, std::back_inserter(topo_data), [&mesh](n_id_t node) {
            return getLocalNodeIndex(mesh, node);
        });
        std::fill_n(std::back_inserter(cell_types), numSubels< ET, EO >(), subelCellType< ET, EO >());
        std::generate_n(std::back_inserter(offsets), numSubels< ET, EO >(), [&offset]() {
            offset += numSubelNodes< ET, EO >();
            return offset;
        });
    };
    mesh.visit(process_element);

    constexpr auto get_b64_alloc_size = []< typename T >(const std::vector< T >& v) {
        return util::getBase64EncodingSize< T >(v.size());
    };
    std::string b64_data;
    b64_data.reserve(get_b64_alloc_size(cell_types) + get_b64_alloc_size(topo_data) + get_b64_alloc_size(offsets));
    const auto encode_b64 = [&]< typename T >(const std::vector< T >& v) {
        const auto enc_size = get_b64_alloc_size(v);
        const auto old_size = b64_data.size();
        b64_data.resize(old_size + enc_size);
        util::encodeAsBase64(v, std::next(begin(b64_data), static_cast< ptrdiff_t >(old_size)));
        return enc_size;
    };

    std::array< size_t, 4 > sizes{};
    auto& [num_cells, topo_sec_sz, offs_sec_sz, type_sec_sz] = sizes;
    num_cells                                                = n_cells;
    topo_sec_sz                                              = encode_b64(topo_data);
    offs_sec_sz                                              = encode_b64(offsets);
    type_sec_sz                                              = encode_b64(cell_types);
    b64_data.shrink_to_fit();
    return std::make_pair(sizes, std::move(b64_data));
}

template < el_o_t... orders >
std::string makeCoordsSerialized(const mesh::MeshPartition< orders... >& mesh)
{
    constexpr size_t space_dim        = 3;
    const auto       n_nodes          = mesh.getAllNodes().size();
    const size_t     n_vals_to_encode = n_nodes * space_dim + 1;
    auto             alloc_to_encode  = util::ArrayOwner< val_t >(n_vals_to_encode);
    const auto       data_to_encode   = std::span{alloc_to_encode};
    const auto       coords           = data_to_encode.subspan(1);

    // First 8 bytes encode the data size in bytes
    data_to_encode.front() = std::bit_cast< val_t >(coords.size_bytes());

    const auto process_element = [&](const auto& element) {
        const auto node_coords = nodePhysicalLocation(element);
        for (size_t i = 0; const auto& point : node_coords)
        {
            const auto local_node_ind = getLocalNodeIndex(mesh, element.getNodes()[i]);
            std::atomic_ref{coords[local_node_ind * space_dim]}.store(point.x(), std::memory_order_relaxed);
            std::atomic_ref{coords[local_node_ind * space_dim + 1]}.store(point.y(), std::memory_order_relaxed);
            std::atomic_ref{coords[local_node_ind * space_dim + 2]}.store(point.z(), std::memory_order_relaxed);
            ++i;
        }
    };
    mesh.visit(process_element, std::execution::par);

    std::string retval;
    retval.resize(util::getBase64EncodingSize< val_t >(n_vals_to_encode));
    util::encodeAsBase64(std::as_const(alloc_to_encode), retval.begin());
    return retval;
}

inline auto encodeFieldImpl(std::span< const val_t > field_vals) -> util::ArrayOwner< char >
{
    const auto n_vals_to_encode = field_vals.size() + 1;
    const auto bytes_encoded    = util::getBase64EncodingSize< val_t >(n_vals_to_encode);
    auto       retval           = util::ArrayOwner< char >(bytes_encoded);
    auto       output_it        = retval.begin();

    // Prepend the size of the results in bytes. We need to create a prefix of size divisible by 3
    auto prefix    = std::array< val_t, 3 >{};
    prefix.front() = std::bit_cast< val_t >(field_vals.size_bytes());
    std::ranges::copy(field_vals | std::views::take(2), std::next(begin(prefix)));

    // Encode prefix (case when field_vals.size() < 2 handled correctly)
    const auto prefix_bytes_written = util::encodeAsBase64(prefix | std::views::take(n_vals_to_encode), output_it);
    std::advance(output_it, prefix_bytes_written);

    // Encode remaining values
    util::encodeAsBase64(field_vals | std::views::drop(2), output_it);
    return retval;
}

template < SizedRangeOfConvertibleTo_c< std::span< const val_t > > Fields >
auto encodeFieldImpl(Fields&& fields) -> util::ArrayOwner< char >
{
    // Move the node value spans into an array for easy random access
    const auto n_fields         = std::ranges::size(fields);
    auto       field_vals_array = std::array< std::span< const val_t >, 3 >{};
    std::ranges::copy(std::forward< Fields >(fields), begin(field_vals_array));
    const auto field_values = std::span{begin(field_vals_array), n_fields};

    constexpr size_t n_fields_to_export = 3; // Always export as 3D, pad 2D results with zeros
    const size_t     n_nodes            = field_values.front().size();
    const size_t     n_field_vals       = n_fields_to_export * n_nodes;
    const size_t     n_vals_to_encode   = n_field_vals + 1;

    auto alloc_to_encode = util::ArrayOwner< val_t >(n_vals_to_encode);
    alloc_to_encode[0]   = std::bit_cast< val_t >(n_field_vals * sizeof(val_t));
    const auto node_vals = std::span{alloc_to_encode}.subspan(1);

    // Interleave field values
    util::tbb::parallelFor(std::views::iota(0ul, n_nodes), [&](size_t node) {
        auto output_iter = std::next(begin(node_vals), static_cast< ptrdiff_t >(node * n_fields_to_export));
        for (auto fv : field_values)
            *output_iter++ = fv[node];
        if (n_fields == 2)
            *output_iter = 0.;
    });

    const auto bytes_encoded = util::getBase64EncodingSize< val_t >(n_vals_to_encode);
    auto       retval        = util::ArrayOwner< char >(bytes_encoded);
    util::encodeAsBase64(std::as_const(alloc_to_encode), retval.begin());
    return retval;
}

template < IndexRange_c Inds >
auto encodeField(const SolutionManager& solution_manager, Inds&& component_inds) -> util::ArrayOwner< char >
{
    const auto n_fields = std::ranges::size(component_inds);
    util::throwingAssert(n_fields != 0 and n_fields <= 3,
                         "Field groupings must have size 1 (scalar), 2 (2D), or 3 (3D)");

    if (n_fields == 1) // Directly encode nodal values
    {
        const auto field_vals = solution_manager.getFieldView(*std::ranges::begin(component_inds));
        return encodeFieldImpl(field_vals);
    }
    else // Values of fields in the group should be interleaved before encoding
        return encodeFieldImpl(std::forward< Inds >(component_inds) |
                               std::views::transform([&solution_manager](size_t component_index) {
                                   return solution_manager.getFieldView(component_index);
                               }));
}

template < SizedRangeOfConvertibleTo_c< std::span< const size_t > > FieldComps >
auto encodeSolution(const SolutionManager& solution_manager,
                    FieldComps&&           field_components) -> std::vector< util::ArrayOwner< char > >
{
    auto retval         = std::vector< util::ArrayOwner< char > >(std::ranges::size(field_components));
    auto grouping_range = std::forward< FieldComps >(field_components) | std::views::common;
    util::tbb::parallelTransform(grouping_range, begin(retval), [&](std::span< const size_t > component_inds) {
        return post::vtk::encodeField(solution_manager, component_inds);
    });
    return retval;
}

inline auto makeNumFieldComponentsView(const util::ArrayOwner< util::ArrayOwner< size_t > >& field_component_inds)
    -> util::ArrayOwner< size_t >
{
    return field_component_inds |
           std::views::transform([](const auto& cmps) -> size_t { return cmps.size() == 1 ? 1 : 3; });
}

inline auto makeEncodedFieldSizeView(const std::vector< util::ArrayOwner< char > >& encoded_fields)
{
    return encoded_fields | std::views::transform([](const auto& enc) { return enc.size(); });
}

inline auto openFileForExport(const std::string& name) -> std::shared_ptr< MpiComm::FileHandle >
{
    const auto open_mode = MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_UNIQUE_OPEN;
    const auto comm_self = MpiComm{MPI_COMM_SELF};
    return std::make_shared< MpiComm::FileHandle >(comm_self.openFile(name.c_str(), open_mode));
}

inline auto makeVtuFileName(std::string_view name, const MpiComm& comm) -> std::string
{
    auto retval = std::filesystem::path{name}.replace_extension().string();
    retval += '_';
    retval += std::to_string(comm.getRank());
    retval += ".vtu";
    return retval;
}

inline auto openVtuFile(std::string_view name, const MpiComm& comm)
{
    return openFileForExport(makeVtuFileName(name, comm));
}

template < std::ranges::range FieldCompInds >
auto makeFieldIndsArrayOwner(FieldCompInds&& inds) -> util::ArrayOwner< util::ArrayOwner< size_t > >
{
    auto retval = util::ArrayOwner< util::ArrayOwner< size_t > >(std::ranges::distance(inds));
    std::ranges::transform(std::forward< FieldCompInds >(inds), retval.begin(), []< typename R >(R&& r) {
        return util::ArrayOwner< size_t >(std::forward< R >(r) | std::views::all);
    });
    return retval;
}
} // namespace post::vtk

template < el_o_t... orders >
PvtuExporter::PvtuExporter(const mesh::MeshPartition< orders... >& mesh) : m_n_nodes{mesh.getAllNodes().size()}
{
    L3STER_PROFILE_FUNCTION;
    updateNodeCoords(mesh);
    initTopo(mesh);
}

template < std::ranges::range FieldCompInds >
void PvtuExporter::exportSolution(std::string_view                       file_name,
                                  const MpiComm&                         comm,
                                  const SolutionManager&                 solution_manager,
                                  const util::ArrayOwner< std::string >& field_names,
                                  FieldCompInds&&                        field_component_inds_view)
    requires RangeOfConvertibleTo_c< std::ranges::range_reference_t< FieldCompInds >, size_t >
{
    L3STER_PROFILE_FUNCTION;
    const auto field_component_inds =
        post::vtk::makeFieldIndsArrayOwner(std::forward< FieldCompInds >(field_component_inds_view));
    util::throwingAssert(field_names.size() == field_component_inds.size(),
                         "Field names and groupings must have the same size");

    std::filesystem::create_directories(std::filesystem::absolute(file_name).parent_path());
    flushWriteQueue();
    if (comm.getRank() == 0)
        enqueuePvtuFileWrite(file_name, comm, field_names, field_component_inds);
    enqueueVtuFileWrite(file_name, comm, solution_manager, field_names, field_component_inds);
}

template < el_o_t... orders >
void PvtuExporter::updateNodeCoords(const mesh::MeshPartition< orders... >& mesh)
{
    flushWriteQueue(); // Async write of previous coords may be in flight
    m_encoded_coords = post::vtk::makeCoordsSerialized(mesh);
}

template < el_o_t... orders >
void PvtuExporter::initTopo(const mesh::MeshPartition< orders... >& mesh)
{
    auto topo_serialization                                       = post::vtk::serializeTopology(mesh);
    auto& [sizes, data]                                           = topo_serialization;
    const auto [num_cells, topo_sec_sz, offs_sec_sz, type_sec_sz] = sizes;
    m_n_cells                                                     = num_cells;
    m_section_sizes.topology                                      = topo_sec_sz;
    m_section_sizes.offsets                                       = offs_sec_sz;
    m_section_sizes.cell_types                                    = type_sec_sz;
    m_encoded_topo                                                = std::move(data);
}

void PvtuExporter::enqueuePvtuFileWrite(std::string_view                                      file_name,
                                        const MpiComm&                                        comm,
                                        const util::ArrayOwner< std::string >&                field_names,
                                        const util::ArrayOwner< util::ArrayOwner< size_t > >& field_component_inds)
{
    const auto n_field_components_view = post::vtk::makeNumFieldComponentsView(field_component_inds);
    auto       pvtu_contents =
        post::vtk::makePvtuFileContents(file_name, comm.getSize(), field_names, n_field_components_view);
    const auto pvtu_name = std::filesystem::path(file_name).replace_extension("pvtu").string();
    auto       pvtu_file = post::vtk::openFileForExport(pvtu_name);
    enqueueWrite(std::move(pvtu_file), 0, std::move(pvtu_contents));
}

void PvtuExporter::enqueueVtuFileWrite(std::string_view                                      file_name,
                                       const MpiComm&                                        comm,
                                       const SolutionManager&                                solution_manager,
                                       const util::ArrayOwner< std::string >&                field_names,
                                       const util::ArrayOwner< util::ArrayOwner< size_t > >& field_component_inds)
{
    const auto file_name_vtu_ext = std::filesystem::path{file_name}.replace_extension("vtu").string();
    const auto vtu_file_handle   = post::vtk::openVtuFile(file_name_vtu_ext, comm);
    auto       enqueue_write     = [this, &vtu_file_handle, pos = MPI_Offset{0}](auto&& text) mutable {
        pos += enqueueWrite(vtu_file_handle, pos, std::forward< decltype(text) >(text));
    };

    auto encoded_fields = post::vtk::encodeSolution(solution_manager, field_component_inds);
    enqueue_write(makeDataDescription(field_names, field_component_inds, encoded_fields));
    enqueue_write(m_encoded_coords);
    enqueue_write(m_encoded_topo);
    for (auto& enc_fld : encoded_fields)
        enqueue_write(std::move(enc_fld));
    enqueue_write(post::vtk::vtu_postamble);
}

auto PvtuExporter::makeDataDescription(const util::ArrayOwner< std::string >&                field_names,
                                       const util::ArrayOwner< util::ArrayOwner< size_t > >& field_component_inds,
                                       const std::vector< util::ArrayOwner< char > >&        encoded_fields) const
    -> std::string
{
    const auto n_field_components  = post::vtk::makeNumFieldComponentsView(field_component_inds);
    auto&&     encoded_field_sizes = post::vtk::makeEncodedFieldSizeView(encoded_fields);

    std::string retval;
    retval.reserve(1u << 12);
    retval += post::vtk::vtu_preamble;
    auto append_data_array =
        [&retval,
         offset = 0uz](std::string_view type, std::string_view name, size_t n_comps, size_t size_bytes) mutable {
            retval += R"(  <DataArray type=")";
            retval += type;
            retval += R"(" Name=")";
            retval += name;
            retval += R"(" NumberOfComponents=")";
            retval += std::to_string(n_comps);
            retval += R"(" format="appended" offset=")";
            retval += std::to_string(offset);
            retval += "\">\n";
            retval += "  </DataArray>\n";
            offset += size_bytes;
        };

    // Mesh description
    retval += R"(<Piece NumberOfPoints=")";
    retval += std::to_string(m_n_nodes);
    retval += R"(" NumberOfCells=")";
    retval += std::to_string(m_n_cells);
    retval += "\">\n<Points>\n";
    append_data_array("Float64", "Position", 3, m_encoded_coords.size());
    retval += "</Points>\n";
    retval += "<Cells>\n";
    append_data_array("UInt32", "connectivity", 1, m_section_sizes.topology);
    append_data_array("UInt32", "offsets", 1, m_section_sizes.offsets);
    append_data_array("UInt8", "types", 1, m_section_sizes.cell_types);
    retval += "</Cells>\n";

    // Data description
    retval += "<PointData";
    if (std::ranges::any_of(n_field_components, [](size_t sz) { return sz == 1; }))
    {
        const auto first_scal     = std::ranges::find(n_field_components, 1);
        const auto first_scal_ind = std::distance(std::ranges::begin(n_field_components), first_scal);
        retval += " Scalars=\"";
        retval += *std::next(std::ranges::begin(field_names), first_scal_ind);
        retval += "\"";
    }
    if (std::ranges::any_of(n_field_components, [](auto sz) { return sz > 1; }))
    {
        const auto first_vec     = std::ranges::find_if(n_field_components, [](auto sz) { return sz > 1; });
        const auto first_vec_ind = std::distance(std::ranges::begin(n_field_components), first_vec);
        retval += " Vectors=\"";
        retval += *std::next(std::ranges::begin(field_names), first_vec_ind);
        retval += "\"";
    }
    retval += ">\n";

    auto sz_it    = std::ranges::begin(encoded_field_sizes);
    auto n_cmp_it = std::ranges::begin(n_field_components);
    for (const auto& name : field_names)
        append_data_array("Float64", name, *n_cmp_it++, *sz_it++);

    retval += "</PointData>\n";
    retval += "<CellData>\n";
    retval += "</CellData>\n";
    retval += "</Piece>\n";
    retval += "</UnstructuredGrid>\n";
    retval += "<AppendedData encoding=\"base64\">\n  _";
    return retval;
}

template < ContiguousSizedRangeOf< char > Text >
MPI_Offset PvtuExporter::enqueueWrite(std::shared_ptr< MpiComm::FileHandle > file, MPI_Offset pos, Text&& text)
    requires std::ranges::borrowed_range< Text > or std::same_as< std::string, Text > or
             std::same_as< util::ArrayOwner< char >, Text >
{
    const MPI_Offset write_length = std::ranges::ssize(text);
    if constexpr (std::ranges::borrowed_range< Text >)
    {
        auto request = file->writeAtAsync(std::forward< Text >(text), pos);
        m_write_queue.emplace_back(std::move(file), std::monostate{}, std::move(request));
    }
    else
    {
        auto request = file->writeAtAsync(text, pos);
        m_write_queue.emplace_back(std::move(file), std::forward< Text >(text), std::move(request));
    }
    return write_length;
}

void PvtuExporter::flushWriteQueue()
{
    MpiComm::Request::waitAll(m_write_queue |
                              std::views::transform([](AsyncWrite& aw) -> MpiComm::Request& { return aw.request; }));
    m_write_queue.clear();
}
} // namespace lstr
#endif // L3STER_POST_VTKEXPORT_HPP
