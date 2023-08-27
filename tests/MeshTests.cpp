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
    auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), gmsh_tag);

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
    auto boundaries = std::vector< BoundaryView< 1 > >{};
    boundaries.reserve(4);
    for (d_id_t i = 2; i <= 5; ++i)
        boundaries.emplace_back(mesh, std::views::single(i));
    CHECK(boundaries[0].size() == 10);
    CHECK(boundaries[1].size() == 10);
    CHECK(boundaries[2].size() == 10);
    CHECK(boundaries[3].size() == 10);
    CHECK(BoundaryView{mesh, std::views::single(6)}.size() == 0);
}

TEST_CASE("3D mesh import", "[mesh]")
{
    auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube.msh), gmsh_tag);

    constexpr size_t expected_nvertices = 5885;
    constexpr size_t expected_nelements = 5664;
    CHECK(mesh.getNElements() == expected_nelements);
    CHECK(mesh.getOwnedNodes().size() == expected_nvertices);

    for (d_id_t i = 2; i <= 7; ++i)
        CHECK_NOTHROW(BoundaryView{mesh, std::views::single(i)});
    CHECK(BoundaryView{mesh, std::views::single(42)}.size() == 0);
}

TEST_CASE("Unsupported mesh formats, mesh I/O error handling", "[mesh]")
{
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii2.msh), gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_bin2.msh), gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_bin4.msh), gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(nonexistent.msh), gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_triangle_mesh.msh), gmsh_tag));
}

TEST_CASE("Incorrect domain dim handling", "[mesh]")
{
    auto domain = Domain< 1 >{};

    auto                    data = ElementData< ElementType::Line, 1 >{{Point{0., 0., 0.}, Point{0., 0., 0.}}};
    std::array< n_id_t, 2 > nodes{1, 2};
    el_id_t                 id = 1;
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
        std::array< double, 20 > retval{};
        std::ranges::generate(retval, [v = 0.]() mutable { return v += 1.; });
        return retval;
    });
    constexpr auto n_edges   = node_dist.size() - 1;
    constexpr auto n_faces   = n_edges * n_edges * 6;
    constexpr auto n_vols    = n_edges * n_edges * n_edges;
    const auto     mesh      = makeCubeMesh(node_dist);
    const auto     policy    = TestType{};

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

    mesh.visit(element_counter, std::array{1, 2, 3, 4, 5, 6, 7}, policy);
    CHECK(count == n_faces);
    count = 0;

    std::as_const(mesh).visit(element_counter, std::array{1, 2, 3, 4, 5, 6}, policy);
    CHECK(count == n_faces);
    count = 0;
}

TEMPLATE_TEST_CASE("Element lookup", "[mesh]", std::execution::sequenced_policy, std::execution::parallel_policy)
{
    const auto policy = TestType{};
    auto       mesh   = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), gmsh_tag);

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

    auto    data  = ElementData< ElementType::Line, 1 >{{Point{0., 0., 0.}, Point{0., 0., 0.}}};
    auto    nodes = std::array< n_id_t, 2 >{1, 2};
    el_id_t id    = 1;
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
    constexpr auto n_parts    = 2u;
    const auto     mesh       = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), gmsh_tag);
    const auto     n_elements = mesh.getNElements();
    REQUIRE_NOTHROW(partitionMesh(mesh, 1, {}));
    const auto partitions = partitionMesh(mesh, n_parts, {2, 3, 4, 5});
    REQUIRE(partitions.size() == n_parts);
    REQUIRE(
        std::transform_reduce(partitions.cbegin(), partitions.cend(), 0u, std::plus{}, [](const MeshPartition< 1 >& p) {
            return p.getNElements();
        }) == n_elements);
    CHECK(partitions.front().getNElements() == Approx(partitions.back().getNElements()).epsilon(.1));
    std::vector< n_id_t > intersects;
    std::ranges::set_intersection(
        partitions.front().getOwnedNodes(), partitions.back().getOwnedNodes(), std::back_inserter(intersects));
    CHECK(intersects.empty());
    intersects.clear();
    std::ranges::set_intersection(
        partitions.front().getGhostNodes(), partitions.back().getGhostNodes(), std::back_inserter(intersects));
    CHECK(intersects.empty());
    for (const auto& part : partitions)
        REQUIRE_THROWS(partitionMesh(part, 42, {}));
}

TEST_CASE("Boundary views in partitioned meshes", "[mesh]")
{
    constexpr auto n_parts          = 8u;
    constexpr auto node_dist        = std::array{0., 1., 2., 3., 4., 5., 6., 7., 8.};
    const auto     mesh             = makeCubeMesh(node_dist);
    auto           partitions       = std::vector< MeshPartition< 1 > >{};
    const auto     check_boundaries = [&] {
        for (auto&& part : partitions)
        {
            const auto domain_ids = part.getDomainIds();
            for (d_id_t boundary = 1; boundary <= 6; ++boundary)
            {
                if (std::ranges::find(domain_ids, boundary) == domain_ids.end())
                    continue;
                const auto boundary_view = BoundaryView{part, std::views::single(boundary)};
                CHECK(part.getDomain(boundary).getNElements() == boundary_view.size());
                size_t boundary_size = 0;
                boundary_view.visit([&](const auto&) { ++std::atomic_ref{boundary_size}; }, std::execution::par);
                CHECK(boundary_size == boundary_view.size());
            }
        }
    };
    SECTION("Boundaries assigned correctly")
    {
        partitions = partitionMesh(mesh, n_parts, {1, 2, 3, 4, 5, 6});
        CHECK_NOTHROW(check_boundaries());
    }

    SECTION("Boundaries assigned incorrectly")
    {
        partitions = partitionMesh(mesh, n_parts, {});
        CHECK_THROWS(check_boundaries());
    }
}

TEST_CASE("Mesh conversion to higher order", "[mesh]")
{
    SECTION("mesh imported from gmsh")
    {
        constexpr el_o_t order      = 2;
        auto             mesh1      = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube.msh), gmsh_tag);
        const auto       n_elements = mesh1.getNElements();

        auto mesh = convertMeshToOrder< order >(mesh1);
        CHECK(n_elements == mesh.getNElements());
        const auto validate_elorder = [&]< ElementType T, el_o_t O >(const Element< T, O >&) {
            if constexpr (O != 2)
                throw std::logic_error{"Incorrect element order"};
        };
        CHECK_NOTHROW(mesh.visit(validate_elorder));
        CHECK(mesh.getOwnedNodes().size() == 44745u);
        for (size_t i = 0; i < mesh.getOwnedNodes().size() - 1; ++i)
            CHECK(mesh.getOwnedNodes()[i] + 1 == mesh.getOwnedNodes()[i + 1]);
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
