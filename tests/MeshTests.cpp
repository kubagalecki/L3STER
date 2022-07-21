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

TEST_CASE("2D mesh import", "[mesh]")
{
    auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), gmsh_tag);

    REQUIRE(mesh.getPartitions().size() == 1);

    auto&            part             = mesh.getPartitions()[0];
    constexpr size_t expected_n_nodes = 121;
    size_t           max_node{};
    const auto       update_max_node = [&](const auto& el) {
        const size_t max_el_node = *std::ranges::max_element(el.getNodes());
        if (max_el_node >= expected_n_nodes)
            throw std::invalid_argument{"max nodes exceeded"};
        max_node = std::max(max_node, max_el_node);
    };
    part.visit(update_max_node);
    REQUIRE(max_node + 1 == expected_n_nodes);
    REQUIRE(max_node + 1 == part.getNodes().size());
    REQUIRE(part.getGhostNodes().size() == 0);

    // BoundaryView
    part.initDualGraph();
    std::vector< BoundaryView > boundaries;
    boundaries.reserve(4);
    for (int i = 2; i <= 5; ++i)
        boundaries.push_back(part.getBoundaryView(i));
    CHECK(boundaries[0].size() == 10);
    CHECK(boundaries[1].size() == 10);
    CHECK(boundaries[2].size() == 10);
    CHECK(boundaries[3].size() == 10);
    CHECK(part.getBoundaryView(6).size() == 0);
}

TEST_CASE("3D mesh import", "[mesh]")
{
    auto  mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube.msh), gmsh_tag);
    auto& part = mesh.getPartitions()[0];
    part.initDualGraph();

    constexpr size_t expected_nvertices = 5885;
    constexpr size_t expected_nelements = 5664;
    CHECK(part.getNElements() == expected_nelements);
    CHECK(part.getNodes().size() == expected_nvertices);

    for (int i = 2; i <= 7; ++i)
        CHECK_NOTHROW(part.getBoundaryView(i));
    CHECK(part.getBoundaryView(42).size() == 0);
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
    Domain d;

    ElementData< ElementTypes::Line, 1 > data{{Point{0., 0., 0.}, Point{0., 0., 0.}}};
    std::array< n_id_t, 2 >              nodes{1, 2};
    el_id_t                              id = 1;
    d.reserve< ElementTypes::Line, 1 >(1);
    d.emplaceBack< ElementTypes::Line, 1 >(nodes, data, id++);
    d.reserve< ElementTypes::Line, 1 >(1);
    CHECK_THROWS(d.reserve< ElementTypes::Quad, 1 >(1));
    CHECK_THROWS(d.getElementVector< ElementTypes::Quad, 1 >());
}

TEMPLATE_TEST_CASE("Iteration over elements",
                   "[mesh]",
                   std::execution::sequenced_policy,
                   std::execution::parallel_policy)
{
    constexpr auto node_dist = [] {
        std::array< double, 20 > retval{};
        std::ranges::generate(retval, [v = 0.]() mutable { return v += 1.; });
        return retval;
    }();
    constexpr auto n_edges = node_dist.size() - 1;
    constexpr auto n_faces = n_edges * n_edges * 6;
    constexpr auto n_vols  = n_edges * n_edges * n_edges;
    auto           mesh    = makeCubeMesh(node_dist);
    auto&          part    = mesh.getPartitions()[0];
    const auto     policy  = TestType{};

    int        count           = 0;
    const auto element_counter = [&](const auto& el) {
        ++std::atomic_ref{count};
    };

    part.visit(element_counter, policy);
    CHECK(count == n_faces + n_vols);
    count = 0;

    std::as_const(part).visit(element_counter, policy);
    CHECK(count == n_faces + n_vols);
    count = 0;

    part.visit([&count](const auto&, DomainView) { ++std::atomic_ref{count}; }, policy);
    CHECK(count == n_faces + n_vols);
    count = 0;

    std::as_const(part).visit([&count](const auto&, DomainView) { ++std::atomic_ref{count}; }, policy);
    CHECK(count == n_faces + n_vols);
    count = 0;

    part.visit(
        element_counter, [](const DomainView& dv) { return dv.getDim() == 2; }, policy);
    CHECK(count == n_faces);
    count = 0;

    std::as_const(part).visit(
        element_counter, [](const DomainView& dv) { return dv.getDim() == 2; }, policy);
    CHECK(count == n_faces);
    count = 0;

    part.visit(
        [&count](const auto&, DomainView dv) {
            ++std::atomic_ref{count};
            CHECK(dv.getDim() == 2);
        },
        [](const DomainView& dv) { return dv.getDim() == 2; },
        policy);
    CHECK(count == n_faces);
    count = 0;

    std::as_const(part).visit(
        [&count](const auto&, DomainView dv) {
            ++std::atomic_ref{count};
            CHECK(dv.getDim() == 2);
        },
        [](const DomainView& dv) { return dv.getDim() == 2; },
        policy);
    CHECK(count == n_faces);
    count = 0;

    part.visit(element_counter, std::array{1, 2, 3, 4, 5, 6, 7}, policy);
    CHECK(count == n_faces);
    count = 0;

    std::as_const(part).visit(element_counter, std::array{1, 2, 3, 4, 5, 6}, policy);
    CHECK(count == n_faces);
    count = 0;

    part.visit(
        [&count](const auto&, DomainView dv) {
            ++std::atomic_ref{count};
            CHECK(dv.getDim() == 2);
        },
        std::array{1, 2, 3, 4, 5, 6, 7},
        policy);
    CHECK(count == n_faces);
    count = 0;

    std::as_const(part).visit(
        [&count](const auto&, DomainView dv) {
            ++std::atomic_ref{count};
            CHECK(dv.getDim() == 2);
        },
        std::array{1, 2, 3, 4, 5, 6},
        policy);
    CHECK(count == n_faces);
    count = 0;
}

TEMPLATE_TEST_CASE("Element lookup", "[mesh]", std::execution::sequenced_policy, std::execution::parallel_policy)
{
    const auto  policy = TestType{};
    const auto  mesh   = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), gmsh_tag);
    const auto& part   = mesh.getPartitions()[0];
    CHECK_THROWS(part.getDualGraph());

    constexpr auto existing_nodes = std::array{54, 55, 63, 64};
    constexpr auto fake_nodes     = std::array{153, 213, 372, 821};

    CHECK(part.find(
        [&](const auto& element) {
            auto element_nodes = element.getNodes();
            std::ranges::sort(element_nodes);
            return std::ranges::equal(element_nodes, existing_nodes);
        },
        policy));
    CHECK(part.find(
        [&](const auto& element) {
            auto element_nodes = element.getNodes();
            std::ranges::sort(element_nodes);
            return std::ranges::equal(element_nodes, existing_nodes);
        },
        [](DomainView dv) { return dv.getDim() == 2; },
        policy));
    CHECK_FALSE(part.find(
        [&](const auto& element) {
            auto element_nodes = element.getNodes();
            std::ranges::sort(element_nodes);
            return std::ranges::equal(element_nodes, fake_nodes);
        },
        policy));
    CHECK_FALSE(MeshPartition{}.find([](const auto& el) { return true; }, policy));
    CHECK_FALSE(MeshPartition{}.find(0));
}

TEST_CASE("Element lookup by ID", "[mesh]")
{
    Domain d;
    CHECK_FALSE(d.find(1));
    CHECK_FALSE(std::as_const(d).find(1));

    ElementData< ElementTypes::Line, 1 > data{{Point{0., 0., 0.}, Point{0., 0., 0.}}};
    std::array< n_id_t, 2 >              nodes{1, 2};
    el_id_t                              id = 1;
    d.reserve< ElementTypes::Line, 1 >(1);
    CHECK_FALSE(d.find(1));
    d.emplaceBack< ElementTypes::Line, 1 >(nodes, data, id++);
    CHECK(d.find(1));
    CHECK_FALSE(d.find(0));

    const auto next = [&]() -> std::array< n_id_t, 2 >& {
        std::ranges::for_each(nodes, [](auto& n) { ++n; });
        return nodes;
    };

    for (size_t i = 0; i < 10; ++i)
        d.emplaceBack< ElementTypes::Line, 1 >(next(), data, id++);
    CHECK(d.find(10));

    d.emplaceBack< ElementTypes::Line, 1 >(next(), data, 0);
    CHECK(d.find(0));

    d.emplaceBack< ElementTypes::Line, 1 >(next(), data, id *= 2);
    CHECK(d.find(id));
    CHECK_FALSE(d.find(id + 1));
}

TEST_CASE("Serial mesh partitioning", "[mesh]")
{
    constexpr auto n_parts    = 2u;
    auto           mesh       = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), gmsh_tag);
    const auto     n_elements = mesh.getPartitions()[0].getNElements();
    REQUIRE_NOTHROW(mesh = partitionMesh(mesh, 1, {}));
    mesh = partitionMesh(mesh, n_parts, {2, 3, 4, 5});
    REQUIRE(mesh.getPartitions().size() == n_parts);
    REQUIRE(std::accumulate(
                mesh.getPartitions().cbegin(), mesh.getPartitions().cend(), 0u, [](size_t size, const auto& part) {
                    return size + part.getNElements();
                }) == n_elements);
    auto& p1 = mesh.getPartitions()[0];
    auto& p2 = mesh.getPartitions()[1];
    CHECK(p1.getNElements() == Approx(p2.getNElements()).epsilon(.1));
    std::vector< n_id_t > intersects;
    std::ranges::set_intersection(p1.getNodes(), p2.getNodes(), std::back_inserter(intersects));
    CHECK(intersects.empty());
    intersects.clear();
    std::ranges::set_intersection(p1.getGhostNodes(), p2.getGhostNodes(), std::back_inserter(intersects));
    CHECK(intersects.empty());
    REQUIRE_THROWS(partitionMesh(mesh, 42, {}));
}

TEST_CASE("Boundary views in partitioned meshes", "[mesh]")
{
    constexpr auto node_dist = std::array{0., 1., 2., 3., 4., 5., 6., 7., 8.};
    auto           mesh      = makeCubeMesh(node_dist);

    constexpr auto n_parts          = 8u;
    const auto     check_boundaries = [&] {
        for (auto&& part : mesh.getPartitions())
        {
            const auto domain_ids = part.getDomainIds();
            for (d_id_t boundary = 1; boundary <= 6; ++boundary)
            {
                if (std::ranges::find(domain_ids, boundary) == domain_ids.end())
                    continue;
                const auto boundary_view = part.getBoundaryView(boundary);
                CHECK(part.getDomain(boundary).getNElements() == boundary_view.size());
                size_t boundary_size = 0;
                boundary_view.visit([&](const auto&) { ++std::atomic_ref{boundary_size}; }, std::execution::par);
                CHECK(boundary_size == boundary_view.size());
            }
        }
    };
    SECTION("Boundaries assigned correctly")
    {
        mesh = partitionMesh(mesh, n_parts, {1, 2, 3, 4, 5, 6});
        CHECK_NOTHROW(check_boundaries());
    }

    SECTION("Boundaries assigned incorrectly")
    {
        mesh = partitionMesh(mesh, n_parts, {});
        CHECK_THROWS(check_boundaries());
    }
}

TEST_CASE("Mesh conversion to higher order", "[mesh]")
{
    SECTION("mesh imported from gmsh")
    {
        constexpr el_o_t order      = 2;
        auto             mesh       = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube.msh), gmsh_tag);
        auto&            part       = mesh.getPartitions()[0];
        const auto       n_elements = part.getNElements();

        CHECK_THROWS(part = convertMeshToOrder< order >(part));

        part.initDualGraph();
        part = convertMeshToOrder< order >(part);
        CHECK(n_elements == part.getNElements());
        const auto validate_elorder = [&]< ElementTypes T, el_o_t O >(const Element< T, O >&) {
            if constexpr (O != 2)
                throw std::logic_error{"Incorrect element order"};
        };
        CHECK_NOTHROW(part.visit(validate_elorder));
        CHECK(part.getNodes().size() == 44745u);
        for (size_t i = 0; i < part.getNodes().size() - 1; ++i)
            CHECK(part.getNodes()[i] + 1 == part.getNodes()[i + 1]);
    }

    SECTION("procedurally generated mesh")
    {
        constexpr el_o_t     order = 2;
        constexpr std::array dist{0., .2, .4, .6, .8, 1.};
        constexpr auto       n_edge_els = dist.size() - 1;
        auto                 mesh       = makeCubeMesh(dist);
        auto&                part       = mesh.getPartitions()[0];
        part.initDualGraph();
        const auto n_elements = part.getNElements();
        const auto n_nodes_o1 = part.getNodes().size();
        REQUIRE(n_elements == n_edge_els * n_edge_els * n_edge_els + 6 * n_edge_els * n_edge_els);
        REQUIRE(n_nodes_o1 == dist.size() * dist.size() * dist.size());

        part = convertMeshToOrder< order >(part);
        CHECK(part.getNElements() == n_elements);
        const auto expected_edge_nodes = (dist.size() - 1) * order + 1;
        const auto expected_n_nodes    = expected_edge_nodes * expected_edge_nodes * expected_edge_nodes;
        CHECK(part.getNodes().size() == expected_n_nodes);
        const auto validate_elorder = [&]< ElementTypes T, el_o_t O >(const Element< T, O >&) {
            if constexpr (O != 2)
                throw std::logic_error{"Incorrect element order"};
        };
        CHECK_NOTHROW(part.visit(validate_elorder));
        for (size_t i = 0; i < part.getNodes().size() - 1; ++i)
            CHECK(part.getNodes()[i] + 1 == part.getNodes()[i + 1]);
    }
}
