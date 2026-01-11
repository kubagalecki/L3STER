#ifndef L3STER_POST_NATIVEIO_HPP
#define L3STER_POST_NATIVEIO_HPP

#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/MeshUtils.hpp"
#include "l3ster/post/SolutionManager.hpp"
#include "l3ster/util/IO.hpp"

#include <filesystem>
#include <print>

namespace lstr
{
template < el_o_t... EO >
void save(const MpiComm&                      comm,
          const mesh::MeshPartition< EO... >& mesh,
          const SolutionManager&              solution_manager,
          const std::filesystem::path&        path,
          const util::ArrayOwner< size_t >&   inds,
          std::string                         comment = {})
{
    util::throwingAssert(not inds.empty());
    util::throwingAssert(std::ranges::max(inds) < solution_manager.nFields());

    const auto num_nodes_local  = mesh.getNodeOwnership().owned().size();
    const auto node_begin       = std::invoke([&] {
        auto num_nodes = num_nodes_local;
        comm.exclusiveScanInPlace(std::span{&num_nodes, 1}, MPI_SUM);
        if (comm.getRank() == 0)
            num_nodes = 0;
        return num_nodes;
    });
    const auto num_nodes_global = std::invoke([&] {
        auto num_nodes = num_nodes_local + node_begin;
        comm.broadcast(std::span{&num_nodes, 1}, comm.getSize() - 1);
        return num_nodes;
    });
    const auto header           = std::invoke([&] {
        std::ranges::replace(comment, '\n', ' ');
        auto retval = std::format("L3STER results file\nv1.0\n// {}\n", comment);
        util::serialize(inds.size(), std::back_inserter(retval));
        util::serialize(num_nodes_global, std::back_inserter(retval));
        return retval;
    });

    const auto file = comm.openFile(path.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR | MPI_MODE_UNIQUE_OPEN);
    file.preallocate(header.size() + inds.size() * sizeof(val_t) * num_nodes_global);
    const auto header_req = comm.getRank() == 0 ? file.writeAtAsync(header, 0) : MpiComm::Request{};
    auto       data_reqs  = inds | std::views::enumerate | std::views::transform([&](auto i_ind) {
                         const auto [i, ind]    = i_ind;
                         const auto data        = solution_manager.getFieldView(ind).first(num_nodes_local);
                         const auto node_ofs    = num_nodes_global * static_cast< size_t >(i) + node_begin;
                         const auto dest_offset = header.size() + sizeof(val_t) * node_ofs;
                         return file.writeAtAsync(data, dest_offset);
                     }) |
                     std::ranges::to< util::ArrayOwner >();
    MpiComm::Request::waitAll(data_reqs);
}

template < el_o_t... EO >
void save(const MpiComm&                      comm,
          const mesh::MeshPartition< EO... >& mesh,
          const SolutionManager&              solution_manager,
          const std::filesystem::path&        path,
          std::string                         comment = {})
{
    save(comm, mesh, solution_manager, path, std::views::iota(0uz, solution_manager.nFields()), std::move(comment));
}

template < el_o_t... EO >
void save(const MpiComm&                      comm,
          const mesh::MeshPartition< EO... >& mesh,
          const std::filesystem::path&        path,
          std::string                         comment = {})
{
    const auto serial_mesh  = mesh::serializeMesh(mesh);
    const auto serial_size  = serial_mesh.size();
    const auto sizes        = std::invoke([&] {
        auto retval = util::ArrayOwner< size_t >(comm.getRank() == 0 ? comm.getSizeUz() : 0uz);
        comm.gather(std::views::single(serial_size), retval.begin(), 0);
        return retval;
    });
    const auto header       = std::invoke([&] {
        using namespace std::string_literals;
        if (comm.getRank())
            return ""s;
        std::ranges::replace(comment, '\n', ' ');
        auto retval = std::format("L3STER mesh file\nv1.0\n// {}\n", comment);
        retval.reserve(retval.size() + (sizes.size() + 1) * sizeof(size_t));
        util::serialize(sizes.size(), std::back_inserter(retval));
        for (auto sz : sizes)
            util::serialize(sz, std::back_inserter(retval));
        return retval;
    });
    const auto write_offset = std::invoke([&] {
        auto offsets = util::ArrayOwner< size_t >(sizes.size());
        std::exclusive_scan(sizes.begin(), sizes.end(), offsets.begin(), header.size());
        size_t retval{};
        comm.scatter(offsets, std::span{&retval, 1}, 0);
        return retval;
    });

    const auto file       = comm.openFile(path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_UNIQUE_OPEN);
    const auto header_req = comm.getRank() == 0 ? file.writeAtAsync(header, 0) : MpiComm::Request{};
    const auto mesh_req   = file.writeAtAsync(serial_mesh, write_offset);
}

namespace post::detail
{
inline auto makeFileAsserter(const std::filesystem::path& path)
{
    return [err_str = std::format("Error parsing results file: {}\n", path.string())](
               bool cond, std::source_location sl = std::source_location::current()) {
        util::throwingAssert(cond, err_str, sl);
    };
}

template < typename Asserter >
auto skipLine(std::string_view& text, const Asserter& asserter)
{
    const auto newline_pos = text.find('\n');
    asserter(newline_pos != std::string_view::npos);
    const auto line = text.substr(0, newline_pos);
    text.remove_prefix(newline_pos + 1);
    return line;
}

class LoadedResults
{
public:
    LoadedResults(const std::filesystem::path& results_file) : m_file{results_file}, m_contents{m_file.view()} // NOLINT
    {
        const auto assert_ok = makeFileAsserter(results_file);
        for (int i = 0; i != 3; ++i) // Skip first 3 lines
            skipLine(m_contents, assert_ok);
        constexpr auto sizes_size = 2 * sizeof(size_t);
        assert_ok(m_contents.size() >= sizes_size);
        const auto [n_fields, n_nodes] = util::deserialize< std::array< size_t, 2 > >(m_contents);
        m_fields                       = n_fields;
        m_nodes                        = n_nodes;
        m_contents.remove_prefix(sizes_size);
    }

    [[nodiscard]] val_t operator()(size_t node, size_t field) const
    {
        const auto offset = sizeof(val_t) * (m_nodes * field + node);
        auto       retval = val_t{};
        std::memcpy(&retval, &m_contents[offset], sizeof(val_t));
        return retval;
    }

    [[nodiscard]] size_t fields() const { return m_fields; }
    [[nodiscard]] size_t nodes() const { return m_nodes; }

private:
    util::MmappedFile m_file;
    std::string_view  m_contents;
    size_t            m_fields{}, m_nodes{};
};

template < typename Asserter >
auto extractSavedPartitionInfo(std::string_view& serial, const Asserter& assert_ok)
    -> std::array< util::ArrayOwner< size_t >, 2 >
{
    for (int i = 0; i != 3; ++i)
        skipLine(serial, assert_ok);
    assert_ok(serial.size() >= sizeof(size_t));
    const auto num_parts = util::deserialize< size_t >(serial);
    serial.remove_prefix(sizeof(size_t));
    auto part_sizes   = std::invoke([&] {
        assert_ok(serial.size() >= num_parts * sizeof(size_t));
        auto retval = util::ArrayOwner< size_t >(num_parts);
        std::memcpy(retval.data(), serial.data(), num_parts * sizeof(size_t));
        serial.remove_prefix(num_parts * sizeof(size_t));
        return retval;
    });
    auto part_offsets = std::invoke([&] {
        auto retval = util::ArrayOwner< size_t >(num_parts);
        std::exclusive_scan(part_sizes.begin(), part_sizes.end(), retval.begin(), 0uz);
        return retval;
    });
    return {std::move(part_offsets), std::move(part_sizes)};
}

template < el_id_t... orders >
auto loadUnifiedMesh(const std::filesystem::path& path) -> mesh::MeshPartition< orders... >
{
    const auto file                       = util::MmappedFile{path};
    auto       contents                   = file.view();
    const auto assert_ok                  = makeFileAsserter(path);
    const auto [part_offsets, part_sizes] = extractSavedPartitionInfo(contents, assert_ok);
    auto dom_map                          = typename mesh::MeshPartition< orders... >::domain_map_t{};
    auto bnd_ids                          = util::ArrayOwner< d_id_t >{};
    for (const auto part : std::views::zip_transform(
             [&](auto ofs, auto sz) {
                 const auto serial = contents.substr(ofs, sz);
                 return mesh::deserializeMesh< orders... >(serial);
             },
             part_offsets,
             part_sizes))
    {
        for (auto dom_id : part.getDomainIds())
        {
            auto& domain = dom_map[dom_id];
            part.visit([&](const auto& element) { mesh::pushToDomain(domain, element); }, dom_id);
        }
        if (bnd_ids.empty())
            bnd_ids = part.getBoundaryIdsCopy();
    }
    return {std::move(dom_map), bnd_ids};
}

template < el_id_t... orders >
auto loadPartitionedMesh(const std::filesystem::path& path, const MpiComm& comm) -> mesh::MeshPartition< orders... >
{
    // const auto permutation = comm::detail::computeOptimizedRankPermutation(comm, )
    const auto file                       = util::MmappedFile{path};
    auto       contents                   = file.view();
    const auto assert_ok                  = makeFileAsserter(path);
    const auto [part_offsets, part_sizes] = extractSavedPartitionInfo(contents, assert_ok);
    util::throwingAssert(part_sizes.size() == comm.getSizeUz(), "Saved mesh has different number of partitions");
    const auto my_ofs = part_offsets.at(comm.getRank());
    const auto my_sz  = part_sizes.at(comm.getRank());
    return mesh::deserializeMesh< orders... >(contents.substr(my_ofs, my_sz));
}
} // namespace post::detail

template < el_id_t... orders >
class Loader
{
public:
    struct Opts
    {
        bool repartition = true;
        bool optimize    = false;
    };

    Loader(std::filesystem::path mesh_path, const Opts& opts = {})
        : m_path{std::move(mesh_path)}, m_opts{opts} {} // NOLINT

    auto loadMesh(const MpiComm& comm) -> std::shared_ptr< mesh::MeshPartition< orders... > >;
    void loadResults(const std::filesystem::path&      results_file,
                     const util::ArrayOwner< size_t >& results_inds,
                     SolutionManager&                  solution_manager,
                     const util::ArrayOwner< size_t >& solman_inds) const
    {
        const auto results = post::detail::LoadedResults{results_file};
        loadResultsImpl(results, results_inds, solution_manager, solman_inds);
    }
    void loadResults(const std::filesystem::path& results_file, SolutionManager& solution_manager) const
    {
        const auto results = post::detail::LoadedResults{results_file};
        loadResultsImpl(results,
                        std::views::iota(0uz, results.fields()),
                        solution_manager,
                        std::views::iota(0uz, solution_manager.nFields()));
    }

private:
    void loadResultsImpl(const post::detail::LoadedResults& results,
                         const util::ArrayOwner< size_t >&  src_inds,
                         SolutionManager&                   solution_manager,
                         const util::ArrayOwner< size_t >&  dest_inds) const;

    auto loadMeshRepart(const MpiComm& comm, const Opts& opts) -> std::shared_ptr< mesh::MeshPartition< orders... > >;
    auto loadMeshDirect(const MpiComm& comm, const Opts& opts) -> std::shared_ptr< mesh::MeshPartition< orders... > >;

    std::filesystem::path      m_path;
    Opts                       m_opts;
    util::ArrayOwner< n_id_t > m_old_nodes;
    n_id_t                     m_max_old_node = std::numeric_limits< n_id_t >::max();
};

template < el_id_t... orders >
auto Loader< orders... >::loadMesh(const MpiComm& comm) -> std::shared_ptr< mesh::MeshPartition< orders... > >
{
    return m_opts.repartition ? loadMeshRepart(comm, m_opts) : loadMeshDirect(comm, m_opts);
}

template < el_id_t... orders >
void Loader< orders... >::loadResultsImpl(const post::detail::LoadedResults& results,
                                          const util::ArrayOwner< size_t >&  src_inds,
                                          SolutionManager&                   solution_manager,
                                          const util::ArrayOwner< size_t >&  dest_inds) const
{
    util::throwingAssert(m_max_old_node != std::numeric_limits< n_id_t >::max(), "You must call loadMesh first");
    util::throwingAssert(src_inds.size() == dest_inds.size());
    util::throwingAssert(not src_inds.empty());
    util::throwingAssert(std::ranges::max(src_inds) < results.fields());
    util::throwingAssert(std::ranges::max(dest_inds) < solution_manager.nFields());
    util::throwingAssert(m_max_old_node < results.nodes());

    const auto dest_view = solution_manager.getRawView();
    for (auto&& [src_ind, dest_ind] : std::views::zip(src_inds, dest_inds))
        for (auto&& [i, n] : m_old_nodes | std::views::enumerate)
            dest_view(i, dest_ind) = results(n, src_ind);
}

template < el_id_t... orders >
auto Loader< orders... >::loadMeshRepart(const MpiComm& comm, const Opts& opts)
    -> std::shared_ptr< mesh::MeshPartition< orders... > >
{
    const auto load_mesh = [&] {
        return post::detail::loadUnifiedMesh< orders... >(m_path);
    };
    auto [mesh, old_node_ids] = comm::distributeMeshAndNodeMap(comm, load_mesh, {}, {.optimize = opts.optimize});
    m_old_nodes               = std::move(old_node_ids);
    m_max_old_node            = m_old_nodes.empty() ? n_id_t{0} : std::ranges::max(m_old_nodes);
    return mesh;
}

template < el_id_t... orders >
auto Loader< orders... >::loadMeshDirect(const MpiComm& comm, const Opts& opts)
    -> std::shared_ptr< mesh::MeshPartition< orders... > >
{
    auto part               = post::detail::loadPartitionedMesh< orders... >(m_path, comm);
    m_old_nodes             = util::ArrayOwner< n_id_t >(part.getNodeOwnership().localSize());
    const auto shared_begin = std::ranges::copy(part.getNodeOwnership().owned(), m_old_nodes.begin()).out;
    std::ranges::copy(part.getNodeOwnership().shared(), shared_begin);
    if (opts.optimize)
        std::tie(part, m_old_nodes) = comm::optimizeMeshDistribution(comm, part, std::span{std::as_const(m_old_nodes)});
    m_max_old_node = m_old_nodes.empty() ? n_id_t{0} : std::ranges::max(m_old_nodes);
    return std::make_shared< mesh::MeshPartition< orders... > >(std::move(part));
}
} // namespace lstr
#endif // L3STER_POST_NATIVEIO_HPP
