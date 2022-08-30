#ifndef L3STER_POST_VTKEXPORT_HPP
#define L3STER_POST_VTKEXPORT_HPP

#include "l3ster/assembly/NodeToDofMap.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"
#include "l3ster/util/Base64.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

#include <string>
#include <string_view>

namespace lstr
{
namespace detail::vtk
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
template < std::ranges::range R >
void appendPvtuPointDataDef(std::string& str, R&& def)
    requires std::convertible_to< std::ranges::range_value_t< R >, std::pair< std::string_view, dim_t > >
{
    bool scal_present = false, vec_present = false; // Only 1 scalar and 1 vector can be present in the header
    for (const auto& [name, n_comps] : def)
    {
        const bool is_vector = n_comps > 1;
        if (is_vector and not std::exchange(vec_present, true))
        {
            str += " Vectors=\"";
            str += name;
            str += '"';
        }
        else if (not scal_present)
        {
            str += " Scalars=\"";
            str += name;
            str += '"';
            scal_present = true;
        }
        if (scal_present and vec_present)
            break;
    }
    str += ">\n";
    for (const auto& [name, n_comps] : def)
    {
        str += R"(  <PDataArray type="Float64" Name=")";
        str += name;
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
        str += R"(<Piece Source=")";
        str += name;
        str += '_';
        str += std::to_string(rank);
        str += ".vtu\"/>\n";
    }
}
template < std::ranges::range R >
std::string makePvtuFileContents(std::string_view name, int n_ranks, R&& def)
    requires std::convertible_to< std::ranges::range_value_t< R >, std::pair< std::string_view, dim_t > >
{
    std::string retval{pvtu_preamble};
    appendPvtuPointDataDef(retval, std::forward< R >(def));
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
template < ElementTypes ET, el_o_t EO >
consteval size_t numSubels()
{
    constexpr auto el_o = static_cast< size_t >(EO);
    if constexpr (ET == ElementTypes::Line)
        return el_o;
    else if constexpr (ET == ElementTypes::Quad)
        return el_o * el_o;
    else if constexpr (ET == ElementTypes::Hex)
        return el_o * el_o * el_o;
    else
        static_assert(ET != ET); // Assert every element type has a corresponding branch
}
template < ElementTypes ET, el_o_t EO >
consteval size_t numSubelNodes()
{
    if constexpr (ET == ElementTypes::Line)
        return 2;
    else if constexpr (ET == ElementTypes::Quad)
        return 4;
    else if constexpr (ET == ElementTypes::Hex)
        return 8;
    else
        static_assert(ET != ET); // Assert every element type has a corresponding branch
}
template < ElementTypes ET, el_o_t EO >
consteval size_t numSerialTopoEntries()
{
    return numSubels< ET, EO >() * numSubelNodes< ET, EO >();
}
template < ElementTypes ET, el_o_t EO >
consteval unsigned char subelCellType()
{
    if constexpr (ET == ElementTypes::Line)
        return 3;
    else if constexpr (ET == ElementTypes::Quad)
        return 9;
    else if constexpr (ET == ElementTypes::Hex)
        return 12;
    else
        static_assert(ET != ET); // Assert every element type has a corresponding branch
}
template < ElementTypes ET, el_o_t EO >
auto serializeElementSubtopo(const Element< ET, EO >& element)
{
    const auto&                                            nodes = element.getNodes();
    std::array< n_id_t, numSerialTopoEntries< ET, EO >() > retval;
    auto                                                   out_topo = retval.begin();
    if constexpr (ET == ElementTypes::Line)
    {
        for (size_t i = 0; i < nodes.size() - 1; ++i)
        {
            *out_topo++ = nodes[i];
            *out_topo++ = nodes[i + 1];
        }
    }
    else if constexpr (ET == ElementTypes::Quad)
    {
        constexpr auto node_per_side = ElementTraits< Element< ElementTypes::Line, EO > >::nodes_per_element;
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
    else if constexpr (ET == ElementTypes::Hex)
    {
        constexpr auto nodes_per_side  = ElementTraits< Element< ElementTypes::Line, EO > >::nodes_per_element;
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
inline std::array< size_t, 2 > getLocalTopoSize(const MeshPartition& mesh)
{
    constexpr auto get_el_entries = []< ElementTypes ET, el_o_t EO >(const Element< ET, EO >&) {
        return std::array{numSubels< ET, EO >(), numSerialTopoEntries< ET, EO >()};
    };
    return mesh.reduce(
        std::array< size_t, 2 >{},
        get_el_entries,
        [](const std::array< size_t, 2 > a1, std::array< size_t, 2 > a2) {
            return std::array< size_t, 2 >{a1[0] + a2[0], a1[1] + a2[1]};
        },
        mesh.getDomainIds());
}
inline unsigned getLocalNodeIndex(const MeshPartition& mesh, n_id_t node)
{
    const auto owned_find_result = std::ranges::lower_bound(mesh.getNodes(), node);
    if (owned_find_result != end(mesh.getNodes()) and *owned_find_result == node)
        return std::distance(begin(mesh.getNodes()), owned_find_result);
    const auto ghost_find_result = std::ranges::lower_bound(mesh.getGhostNodes(), node);
    return mesh.getNodes().size() + std::distance(begin(mesh.getGhostNodes()), ghost_find_result);
}
inline auto serializeTopology(const MeshPartition& mesh)
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
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
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
        return getBase64EncodingSize< T >(v.size());
    };
    std::string b64_data;
    b64_data.reserve(get_b64_alloc_size(cell_types) + get_b64_alloc_size(topo_data) + get_b64_alloc_size(offsets));
    const auto encode_b64 = [&]< typename T >(const std::vector< T >& v) {
        const auto enc_size = get_b64_alloc_size(v);
        const auto old_size = b64_data.size();
        b64_data.resize(old_size + enc_size);
        encodeAsBase64(v, std::next(begin(b64_data), static_cast< ptrdiff_t >(old_size)));
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
template < size_t n_fields >
auto makeNodeDofInfo(const MeshPartition&                            mesh,
                     const NodeToDofMap< n_fields >&                 node_dof_map,
                     const Tpetra::Map< local_dof_t, global_dof_t >& local_global_map)
{
    std::vector< unsigned char > node_dof_vec;
    std::vector< local_dof_t >   dof_local_inds;
    std::vector< unsigned >      node_dof_offsets;
    node_dof_vec.reserve(local_global_map.getNodeNumElements());
    dof_local_inds.reserve(local_global_map.getNodeNumElements());
    node_dof_offsets.reserve(mesh.getNodes().size() + mesh.getGhostNodes().size() + 1);
    node_dof_offsets.push_back(0);

    const auto process_node = [&](n_id_t global_node) {
        const auto& node_dofs   = node_dof_map(global_node);
        unsigned    n_node_dofs = 0;
        for (unsigned char dof_ind = 0; auto dof : node_dofs)
        {
            if (dof != std::numeric_limits< decltype(dof) >::max())
            {
                node_dof_vec.push_back(dof_ind);
                dof_local_inds.push_back(local_global_map.getLocalElement(dof));
                ++n_node_dofs;
            }
            ++dof_ind;
        }
        node_dof_offsets.push_back(node_dof_offsets.back() + n_node_dofs);
    };

    for (auto node : mesh.getNodes())
        process_node(node);
    for (auto node : mesh.getGhostNodes())
        process_node(node);

    node_dof_vec.shrink_to_fit();
    dof_local_inds.shrink_to_fit();
    return std::make_tuple(std::move(node_dof_vec), std::move(dof_local_inds), std::move(node_dof_offsets));
}
inline std::string makeCoordsSerialized(const MeshPartition& mesh)
{
    std::vector< val_t > coords((mesh.getNodes().size() + mesh.getGhostNodes().size()) * 3 + 1);
    // First 8 bytes encode the data size in bytes
    coords[0]                  = std::bit_cast< val_t >((coords.size() - 1u) * sizeof(val_t));
    const auto process_element = [&](const auto& element) {
        const auto node_coords = nodePhysicalLocation(element);
        for (size_t i = 0; const auto& point : node_coords)
        {
            const auto local_node_ind = getLocalNodeIndex(mesh, element.getNodes()[i]);
            std::atomic_ref{coords[local_node_ind * 3 + 1]}.store(point.x(), std::memory_order_relaxed);
            std::atomic_ref{coords[local_node_ind * 3 + 2]}.store(point.y(), std::memory_order_relaxed);
            std::atomic_ref{coords[local_node_ind * 3 + 3]}.store(point.z(), std::memory_order_relaxed);
            ++i;
        }
    };
    mesh.visit(process_element, std::execution::par);

    std::string retval;
    retval.resize(getBase64EncodingSize< val_t >(coords.size()));
    encodeAsBase64(coords, retval.begin());
    return retval;
}
} // namespace detail::vtk

class PvtuExporter
{
public:
    template < size_t dofs_per_node >
    PvtuExporter(const MeshPartition&                              mesh,
                 const NodeToDofMap< dofs_per_node >&              node_dof_map,
                 const Tpetra::Map< local_dof_t, global_dof_t >&   local_global_map,
                 std::vector< std::string >                        field_names,
                 const std::array< unsigned char, dofs_per_node >& field_grouping);

    inline void exportResults(std::string_view                                          name,
                              const MpiComm&                                            comm,
                              const Tpetra::Vector< val_t, local_dof_t, global_dof_t >& sim_values);
    void updateCoords(const MeshPartition& mesh) { m_serialized_coords = detail::vtk::makeCoordsSerialized(mesh); }

private:
    struct AsyncWriteRequest
    {
        std::shared_ptr< MpiComm::FileHandle > file_handle;
        std::string                            content;
        MpiComm::Request                       request;
    };

    inline auto initTopo(const MeshPartition& mesh) -> std::array< size_t, 3 >;
    inline void initFieldSizes();
    inline void initDataSectionOffsets(const std::array< size_t, 3 >& topo_sizes);
    inline void initFieldGroupingInds();
    template < size_t dofs_per_node >
    void initNodeData(const MeshPartition&                            mesh,
                      const NodeToDofMap< dofs_per_node >&            node_dof_map,
                      const Tpetra::Map< local_dof_t, global_dof_t >& local_global_map);

    inline auto unpackForEncoding(std::span< const val_t > values) const -> std::vector< std::vector< val_t > >;
    inline auto encodeData(std::vector< std::vector< val_t > >&& unpacked) const -> std::vector< std::string >;
    inline auto makeDataDescription() const -> std::string;
    inline void doExport(std::string_view name, const MpiComm& comm, std::vector< std::string >&& encoded_data);

    inline void enqueueWrite(std::shared_ptr< MpiComm::FileHandle > file, ptrdiff_t pos, const std::string& text);
    inline void enqueueWrite(std::shared_ptr< MpiComm::FileHandle > file, ptrdiff_t pos, std::string&& text);
    inline void flushWriteQueue();

    size_t                           m_n_cells, m_n_nodes, m_dofs_per_node;
    std::string                      m_serialized_topo, m_serialized_coords;
    std::vector< unsigned char >     m_node_dofs, m_field_sizes, m_field_grouping, m_field_group_inds;
    std::vector< unsigned >          m_node_dof_offsets;
    std::vector< local_dof_t >       m_node_local_dofs;
    std::vector< size_t >            m_data_section_offsets;
    std::vector< std::string >       m_field_names;
    std::vector< AsyncWriteRequest > m_write_queue;
};

template < size_t dofs_per_node >
PvtuExporter::PvtuExporter(const MeshPartition&                              mesh,
                           const NodeToDofMap< dofs_per_node >&              node_dof_map,
                           const Tpetra::Map< local_dof_t, global_dof_t >&   local_global_map,
                           std::vector< std::string >                        field_names,
                           const std::array< unsigned char, dofs_per_node >& field_grouping)
    : m_n_nodes{mesh.getNodes().size() + mesh.getGhostNodes().size()},
      m_dofs_per_node{dofs_per_node},
      m_field_grouping(field_grouping.begin(), field_grouping.end()),
      m_field_names{std::move(field_names)}
{
    updateCoords(mesh);
    const auto topo_sizes = initTopo(mesh);
    initFieldSizes();
    initDataSectionOffsets(topo_sizes);
    initNodeData(mesh, node_dof_map, local_global_map);
    initFieldGroupingInds();
}

auto PvtuExporter::initTopo(const MeshPartition& mesh) -> std::array< size_t, 3 >
{
    auto topo_serialization                                        = detail::vtk::serializeTopology(mesh);
    auto& [sizes, data]                                            = topo_serialization;
    const auto& [num_cells, topo_sec_sz, offs_sec_sz, type_sec_sz] = sizes;
    m_n_cells                                                      = num_cells;
    m_serialized_topo                                              = std::move(data);
    return std::array{topo_sec_sz, offs_sec_sz, type_sec_sz};
}

void PvtuExporter::initFieldSizes()
{
    m_field_sizes.assign(m_field_names.size(), 0);
    for (auto fg : m_field_grouping)
        ++m_field_sizes[fg];
}

void PvtuExporter::initDataSectionOffsets(const std::array< size_t, 3 >& topo_sizes)
{
    static_assert(sizeof(std::uint64_t) == sizeof(val_t)); // we're encoding the number of bytes as val_t
    m_data_section_offsets.reserve(m_field_names.size() + 4);
    m_data_section_offsets.push_back(m_serialized_coords.size());
    for (auto tsz : topo_sizes)
        m_data_section_offsets.push_back(tsz);
    for (auto sz : m_field_sizes)
        m_data_section_offsets.push_back(getBase64EncodingSize< val_t >(sz * m_n_nodes + 1));
    std::exclusive_scan(begin(m_data_section_offsets), end(m_data_section_offsets), begin(m_data_section_offsets), 0ul);
}

template < size_t dofs_per_node >
void PvtuExporter::initNodeData(const MeshPartition&                            mesh,
                                const NodeToDofMap< dofs_per_node >&            node_dof_map,
                                const Tpetra::Map< local_dof_t, global_dof_t >& local_global_map)
{
    auto [node_dofs, local_dofs, offsets] = detail::vtk::makeNodeDofInfo(mesh, node_dof_map, local_global_map);
    m_node_dofs                           = std::move(node_dofs);
    m_node_local_dofs                     = std::move(local_dofs);
    m_node_dof_offsets                    = std::move(offsets);
}

void PvtuExporter::initFieldGroupingInds()
{
    std::unordered_map< unsigned char, unsigned char > group_inds;
    m_field_group_inds.reserve(m_dofs_per_node);
    for (auto group : m_field_grouping)
        m_field_group_inds.push_back(group_inds[group]++);
}

void PvtuExporter::exportResults(std::string_view                                          name,
                                 const MpiComm&                                            comm,
                                 const Tpetra::Vector< val_t, local_dof_t, global_dof_t >& sim_values)
{
    const auto local_values      = sim_values.getData();
    const auto local_values_span = std::span{local_values.begin(), local_values.end()};

    auto unpacked_vals = unpackForEncoding(local_values_span);
    auto encoded_vals  = encodeData(std::move(unpacked_vals));
    doExport(name, comm, std::move(encoded_vals));
}

auto PvtuExporter::makeDataDescription() const -> std::string
{
    std::string retval;
    const auto  append_data_array =
        [&](std::string_view type, std::string_view name, size_t n_comps, std::size_t offset) {
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
        };

    // Mesh description
    retval += R"(<Piece NumberOfPoints=")";
    retval += std::to_string(m_n_nodes);
    retval += R"(" NumberOfCells=")";
    retval += std::to_string(m_n_cells);
    retval += "\">\n<Points>\n";
    append_data_array("Float64", "Position", 3, m_data_section_offsets[0]);
    retval += "</Points>\n";
    retval += "<Cells>\n";
    append_data_array("UInt32", "connectivity", 1, m_data_section_offsets[1]);
    append_data_array("UInt32", "offsets", 1, m_data_section_offsets[2]);
    append_data_array("UInt8", "types", 1, m_data_section_offsets[3]);
    retval += "</Cells>\n";

    // Data description
    retval += "<PointData";
    if (std::ranges::any_of(m_field_sizes, [](auto sz) { return sz == 1; }))
    {
        const auto first_scal     = std::ranges::find(m_field_sizes, 1);
        const auto first_scal_ind = std::distance(begin(m_field_sizes), first_scal);
        retval += " Scalars=\"";
        retval += m_field_names[first_scal_ind];
        retval += "\"";
    }
    if (std::ranges::any_of(m_field_sizes, [](auto sz) { return sz > 1; }))
    {
        const auto first_vec     = std::ranges::find_if(m_field_sizes, [](auto sz) { return sz > 1; });
        const auto first_vec_ind = std::distance(begin(m_field_sizes), first_vec);
        retval += " Vectors=\"";
        retval += m_field_names[first_vec_ind];
        retval += "\"";
    }
    retval += ">\n";
    for (size_t i = 0; const auto& name : m_field_names)
    {
        append_data_array("Float64", name, m_field_sizes[i], m_data_section_offsets[i + 4]);
        ++i;
    }
    retval += "</PointData>\n";
    retval += "<CellData>\n";
    retval += "</CellData>\n";
    retval += "</Piece>\n";
    retval += "</UnstructuredGrid>\n";
    retval += "<AppendedData encoding=\"base64\">\n  _";
    return retval;
}

void PvtuExporter::doExport(std::string_view name, const MpiComm& comm, std::vector< std::string >&& encoded_data)
{
    flushWriteQueue();
    const auto open_mode = MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_UNIQUE_OPEN;
    const auto comm_self = MpiComm{MPI_COMM_SELF};

    // Write the .pvtu file
    if (comm.getRank() == 0)
    {
        // std::views::zip_view is in C++23...
        std::vector< std::pair< std::string_view, dim_t > > def;
        def.reserve(m_field_names.size());
        std::ranges::transform(m_field_names, m_field_sizes, std::back_inserter(def), [](const auto& n, auto s) {
            return std::pair< std::string_view, dim_t >{n, s};
        });
        auto        pvtu_contents = detail::vtk::makePvtuFileContents(name, comm.getSize(), def);
        std::string pvtu_name{name};
        pvtu_name += ".pvtu";
        auto pvtu_file = std::make_shared< MpiComm::FileHandle >(comm_self.openFile(pvtu_name.c_str(), open_mode));
        enqueueWrite(std::move(pvtu_file), 0, std::move(pvtu_contents));
    }

    // Write the .vtu file
    std::string vtu_name{name};
    vtu_name += '_';
    vtu_name += std::to_string(comm.getRank());
    vtu_name += ".vtu";
    const auto vtu_file = std::make_shared< MpiComm::FileHandle >(comm_self.openFile(vtu_name.c_str(), open_mode));

    ptrdiff_t  file_pos  = 0; // manage manually - not sure how the FS file pointer position behaves with async writes
    const auto enq_write = [&]< typename T >(T && content)
        requires std::same_as< std::string, std::decay_t< T > >
    {
        const auto write_size = static_cast< ptrdiff_t >(content.size());
        enqueueWrite(vtu_file, file_pos, std::forward< T >(content));
        file_pos += write_size;
    };
    enq_write(std::string{detail::vtk::vtu_preamble});
    enq_write(makeDataDescription());
    enq_write(m_serialized_coords);
    enq_write(m_serialized_topo);
    for (std::string& d : encoded_data)
        enq_write(std::move(d));
    enq_write(std::string{detail::vtk::vtu_postamble});
}

auto PvtuExporter::unpackForEncoding(std::span< const val_t > values) const -> std::vector< std::vector< val_t > >
{
    std::vector< std::vector< val_t > > fields_for_export(m_field_names.size());
    for (size_t i = 0; auto& field_data : fields_for_export)
    {
        field_data.resize(m_field_sizes[i++] * m_n_nodes + 1, 0.);
        field_data[0] = std::bit_cast< val_t >((field_data.size() - 1u) * sizeof(val_t));
    }

    const auto process_node = [&](unsigned local_node_ind) {
        const auto dofs_begin     = m_node_dof_offsets[local_node_ind];
        const auto dofs_end       = m_node_dof_offsets[local_node_ind + 1];
        const auto dof_inds       = std::span{std::next(begin(m_node_dofs), dofs_begin), dofs_end - dofs_begin};
        const auto local_val_inds = std::span{std::next(begin(m_node_local_dofs), dofs_begin), dof_inds.size()};
        for (size_t i = 0; auto dof_ind : dof_inds)
        {
            const auto field_ind            = m_field_grouping[dof_ind];
            const auto field_vals_begin_ind = local_node_ind * m_field_sizes[field_ind];
            const auto slot                 = m_field_group_inds[dof_ind];

            fields_for_export[field_ind][field_vals_begin_ind + slot + 1] = values[local_val_inds[i++]];
        }
    };
    const auto node_range = tbb::blocked_range< unsigned >(0, m_n_nodes);
    tbb::parallel_for(node_range, [&](const tbb::blocked_range< unsigned >& range) {
        std::ranges::for_each(std::views::iota(range.begin(), range.end()), process_node);
    });
    return fields_for_export;
}

auto PvtuExporter::encodeData(std::vector< std::vector< val_t > >&& unpacked) const -> std::vector< std::string >
{
    std::vector< std::string > retval(unpacked.size());
    const auto                 encode_field = [&](const std::vector< val_t >& src) {
        std::string target;
        target.resize(getBase64EncodingSize< val_t >(src.size()));
        encodeAsBase64(src, begin(target));
        return target;
    };
    std::transform(std::execution::par, begin(unpacked), end(unpacked), begin(retval), encode_field);
    unpacked.clear();
    return retval;
}

void PvtuExporter::enqueueWrite(std::shared_ptr< MpiComm::FileHandle > file, ptrdiff_t pos, const std::string& text)
{
    auto request = file->writeAtAsync(text, pos);
    m_write_queue.emplace_back(std::move(file), text, std::move(request));
}

void PvtuExporter::enqueueWrite(std::shared_ptr< MpiComm::FileHandle > file, ptrdiff_t pos, std::string&& text)
{
    auto request = file->writeAtAsync(text, pos);
    m_write_queue.emplace_back(std::move(file), std::move(text), std::move(request));
}

void PvtuExporter::flushWriteQueue()
{
    m_write_queue.clear();
}
} // namespace lstr
#endif // L3STER_POST_VTKEXPORT_HPP
