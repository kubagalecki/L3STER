#include "l3ster/mesh/BoundaryView.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

#include <algorithm>
#include <atomic>
#include <vector>

using namespace lstr;
using namespace lstr::mesh;

TEST_CASE("2D mesh import", "[mesh]")
{
    constexpr auto boundary_ids = util::makeIotaArray< d_id_t, 4 >(2);
    auto           mesh         = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), boundary_ids, gmsh_tag);

    constexpr size_t expected_n_nodes = 121;
    size_t           max_node{};
    const auto       update_max_node = [&](const auto& el) {
        const size_t max_el_node = *std::ranges::max_element(el.getNodes());
        if (max_el_node >= expected_n_nodes)
            throw std::invalid_argument{"max nodes exceeded"};
        max_node = std::max(max_node, max_el_node);
    };
    mesh.visit(update_max_node);
    REQUIRE(max_node + 1 == expected_n_nodes);
    REQUIRE(max_node + 1 == mesh.getOwnedNodes().size());
    REQUIRE(mesh.getGhostNodes().size() == 0);

    // BoundaryView
    for (auto bnd_id : boundary_ids)
        CHECK(mesh.getBoundary(bnd_id).size() == 10);
    CHECK_THROWS(mesh.getBoundary(boundary_ids.back() + 1));
    CHECK(static_cast< size_t >(std::ranges::distance(mesh.getBoundaryIdsView())) == boundary_ids.size());
}

TEST_CASE("3D mesh import", "[mesh]")
{
    constexpr auto boundary_ids = util::makeIotaArray< d_id_t, 6 >(2);
    auto           mesh         = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube.msh), boundary_ids, gmsh_tag);

    constexpr size_t expected_nvertices = 5885;
    constexpr size_t expected_nelements = 5664;
    CHECK(mesh.getNElements() == expected_nelements);
    CHECK(mesh.getOwnedNodes().size() == expected_nvertices);

    for (d_id_t id : boundary_ids)
        CHECK_NOTHROW(mesh.getBoundary(id));
    CHECK(static_cast< size_t >(std::ranges::distance(mesh.getBoundaryIdsView())) == boundary_ids.size());
}

TEST_CASE("Unsupported mesh formats, mesh I/O error handling", "[mesh]")
{
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii2.msh), {}, gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_bin2.msh), {}, gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_bin4.msh), {}, gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(nonexistent.msh), {}, gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_triangle_mesh.msh), {}, gmsh_tag));
}

TEST_CASE("Incorrect domain dim handling", "[mesh]")
{
    auto domain = Domain< 1 >{};

    const auto     data  = ElementData< ElementType::Line, 1 >{{Point{0., 0., 0.}, Point{0., 0., 0.}}};
    constexpr auto nodes = std::array< n_id_t, 2 >{1, 2};
    el_id_t        id    = 1;
    domain.getElementVector< ElementType::Line, 1 >().reserve(1);
    domain.getElementVector< ElementType::Line, 1 >().emplace_back(nodes, data, id++);
    domain.getElementVector< ElementType::Line, 1 >().reserve(1);
    CHECK_THROWS(domain.getElementVector< ElementType::Quad, 1 >());
}

TEMPLATE_TEST_CASE("Iteration over elements",
                   "[mesh]",
                   std::execution::sequenced_policy,
                   std::execution::parallel_policy)
{
    constexpr auto node_dist = std::invoke([] {
#ifdef NDEBUG
        constexpr auto sz = 20;
#else
        constexpr auto sz = 10;
#endif
        std::array< double, sz > retval{};
        std::ranges::generate(retval, [v = 0.]() mutable { return v += 1.; });
        return retval;
    });
    constexpr auto n_edges = node_dist.size() - 1;
    constexpr auto n_faces = n_edges * n_edges * 6;
    constexpr auto n_vols  = n_edges * n_edges * n_edges;
    const auto     mesh    = makeCubeMesh(node_dist);
    const auto     policy  = TestType{};

    int        count           = 0;
    const auto element_counter = [&](const auto&) {
        ++std::atomic_ref{count};
    };

    mesh.visit(element_counter, policy);
    CHECK(count == n_faces + n_vols);
    count = 0;

    std::as_const(mesh).visit(element_counter, policy);
    CHECK(count == n_faces + n_vols);
    count = 0;

    mesh.visit(element_counter, {1, 2, 3, 4, 5, 6, 7}, policy);
    CHECK(count == n_faces);
    count = 0;

    std::as_const(mesh).visit(element_counter, {1, 2, 3, 4, 5, 6}, policy);
    CHECK(count == n_faces);
    count = 0;

    std::as_const(mesh).visitBoundaries(element_counter, {1, 2, 3, 4, 5, 6, 7}, policy);
    CHECK(count == n_faces);
    count = 0;
}

TEMPLATE_TEST_CASE("Element lookup", "[mesh]", std::execution::sequenced_policy, std::execution::parallel_policy)
{
    const auto policy = TestType{};
    auto       mesh   = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), {}, gmsh_tag);

    constexpr auto existing_nodes = std::array{54, 55, 63, 64};
    constexpr auto fake_nodes     = std::array{153, 213, 372, 821};

    CHECK(mesh.find(
        [&](const auto& element) {
            auto element_nodes = element.getNodes();
            std::ranges::sort(element_nodes);
            return std::ranges::equal(element_nodes, existing_nodes);
        },
        policy));
    CHECK_FALSE(mesh.find(
        [&](const auto& element) {
            auto element_nodes = element.getNodes();
            std::ranges::sort(element_nodes);
            return std::ranges::equal(element_nodes, fake_nodes);
        },
        policy));
    CHECK_FALSE(MeshPartition< 1 >{}.find([](const auto&) { return true; }, policy));
    CHECK_FALSE(MeshPartition< 1 >{}.find(0));
}

TEST_CASE("Element lookup by ID", "[mesh]")
{
    auto domain = Domain< 1 >{};
    CHECK_FALSE(domain.find(1));
    CHECK_FALSE(std::as_const(domain).find(1));

    const auto data  = ElementData< ElementType::Line, 1 >{{Point{0., 0., 0.}, Point{0., 0., 0.}}};
    auto       nodes = std::array< n_id_t, 2 >{1, 2};
    el_id_t    id    = 1;
    domain.getElementVector< ElementType::Line, 1 >().reserve(1);
    CHECK_FALSE(domain.find(1));
    domain.getElementVector< ElementType::Line, 1 >().emplace_back(nodes, data, id++);
    CHECK(domain.find(1));
    CHECK_FALSE(domain.find(0));

    const auto next = [&]() -> std::array< n_id_t, 2 >& {
        std::ranges::for_each(nodes, [](auto& n) { ++n; });
        return nodes;
    };

    for (size_t i = 0; i < 10; ++i)
        domain.getElementVector< ElementType::Line, 1 >().emplace_back(next(), data, id++);
    CHECK(domain.find(10));

    domain.getElementVector< ElementType::Line, 1 >().emplace_back(next(), data, 0);
    CHECK_FALSE(domain.find(0));
    domain.sort();
    CHECK(domain.find(0));

    domain.getElementVector< ElementType::Line, 1 >().emplace_back(next(), data, id *= 2);
    CHECK(domain.find(id));
    CHECK_FALSE(domain.find(id + 1));
}

TEST_CASE("Serial mesh partitioning", "[mesh]")
{
    constexpr auto boundary_ids = std::array< d_id_t, 4 >{2, 3, 4, 5};
    constexpr auto n_parts      = 2;
    const auto     mesh         = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), boundary_ids, gmsh_tag);
    const auto     n_elements   = mesh.getNElements();
    const auto     n_nodes      = mesh.getAllNodes().size();

    // No elements lost
    REQUIRE_NOTHROW(partitionMesh(mesh, 1));
    const auto partitions = partitionMesh(mesh, n_parts);
    REQUIRE(partitions.size() == n_parts);
    REQUIRE(std::transform_reduce(partitions.begin(), partitions.end(), 0u, std::plus{}, [](const auto& part) {
                return part.getNElements();
            }) == n_elements);

    // Partitioning quality acceptable
    CHECK(partitions.front().getNElements() == Approx(partitions.back().getNElements()).epsilon(.1));

    // Nodes partitioned correctly
    CHECK(partitions.front().getOwnedNodes().size() + partitions.back().getOwnedNodes().size() == n_nodes);
    std::vector< n_id_t > intersects;
    std::ranges::set_intersection(
        partitions.front().getOwnedNodes(), partitions.back().getOwnedNodes(), std::back_inserter(intersects));
    CHECK(intersects.empty());
    intersects.clear();
    std::ranges::set_intersection(
        partitions.front().getGhostNodes(), partitions.back().getGhostNodes(), std::back_inserter(intersects));
    CHECK(intersects.empty());

    // Boundaries reconstructed correctly
    constexpr auto get_bnd_size = [](const MeshPartition< 1 >& part, d_id_t bnd_id) -> size_t {
        try
        {
            return part.getBoundary(bnd_id).size();
        }
        catch (const std::out_of_range&)
        {
            return 0;
        }
    };
    for (d_id_t bnd_id : boundary_ids)
        CHECK(get_bnd_size(partitions.front(), bnd_id) + get_bnd_size(partitions.back(), bnd_id) ==
              get_bnd_size(mesh, bnd_id));

    // Check throws on repartitioning
    for (const auto& part : partitions)
        REQUIRE_THROWS(partitionMesh(part, 42));
}

TEST_CASE("Mesh conversion to higher order", "[mesh]")
{
    SECTION("mesh imported from gmsh")
    {
        constexpr el_o_t order = 2;
        auto             mesh1 = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube.msh), {}, gmsh_tag);

        auto mesh = convertMeshToOrder< order >(mesh1);
        CHECK(mesh1.getNElements() == mesh.getNElements());
        const auto validate_elorder = [&]< ElementType T, el_o_t O >(const Element< T, O >&) {
            CHECK(O == 2);
        };
        mesh.visit(validate_elorder);
        CHECK(mesh.getOwnedNodes().size() == 44745u);
        CHECK(std::ranges::adjacent_find(mesh.getOwnedNodes(), [](auto n1, auto n2) { return n2 - n1 != 1; }) ==
              mesh.getOwnedNodes().end());
    }

    SECTION("procedurally generated mesh")
    {
        constexpr el_o_t     order = 2;
        constexpr std::array dist{0., .2, .4, .6, .8, 10.};
        constexpr auto       n_edge_els = dist.size() - 1;
        auto                 mesh1      = makeCubeMesh(dist);
        const auto           n_elements = mesh1.getNElements();
        const auto           n_nodes_o1 = mesh1.getOwnedNodes().size();
        REQUIRE(n_elements == n_edge_els * n_edge_els * n_edge_els + 6 * n_edge_els * n_edge_els);
        REQUIRE(n_nodes_o1 == dist.size() * dist.size() * dist.size());

        auto mesh = convertMeshToOrder< order >(mesh1);
        CHECK(mesh.getNElements() == n_elements);
        const auto expected_edge_nodes = (dist.size() - 1) * order + 1;
        const auto expected_n_nodes    = expected_edge_nodes * expected_edge_nodes * expected_edge_nodes;
        CHECK(mesh.getOwnedNodes().size() == expected_n_nodes);
        for (size_t i = 0; i < mesh.getOwnedNodes().size() - 1; ++i)
            CHECK(mesh.getOwnedNodes()[i] + 1 == mesh.getOwnedNodes()[i + 1]);
    }
}
