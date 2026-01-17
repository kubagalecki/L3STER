#include "l3ster/mesh/BoundaryView.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/LocalMeshView.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"

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
        const size_t max_el_node = *std::ranges::max_element(el.nodes);
        if (max_el_node >= expected_n_nodes)
            throw std::invalid_argument{"max nodes exceeded"};
        max_node = std::max(max_node, max_el_node);
    };
    mesh.visit(update_max_node);
    REQUIRE(max_node + 1 == expected_n_nodes);
    REQUIRE(max_node + 1 == mesh.getNodeOwnership().owned().size());
    REQUIRE(mesh.getNodeOwnership().shared().size() == 0);

    // BoundaryView
    for (auto bnd_id : boundary_ids)
        CHECK(mesh.getBoundary(bnd_id).element_views.size() == 10);
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
    CHECK(mesh.getNodeOwnership().owned().size() == expected_nvertices);

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
    domain.elements.getVector< Element< ElementType::Line, 1 > >().reserve(1);
    domain.elements.getVector< Element< ElementType::Line, 1 > >().emplace_back(nodes, data, id++);
    domain.elements.getVector< Element< ElementType::Line, 1 > >().reserve(1);
    CHECK(domain.elements.getVector< Element< ElementType::Quad, 1 > >().size() == 0);
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

TEST_CASE("Element lookup", "[mesh]")
{
    auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), {}, gmsh_tag);

    constexpr auto existing_nodes = std::array{54, 55, 63, 64};
    constexpr auto fake_nodes     = std::array{153, 213, 372, 821};

    CHECK(mesh.find([&](const auto& element) {
        auto element_nodes = element.nodes;
        std::ranges::sort(element_nodes);
        return std::ranges::equal(element_nodes, existing_nodes);
    }));
    CHECK_FALSE(mesh.find([&](const auto& element) {
        auto element_nodes = element.nodes;
        std::ranges::sort(element_nodes);
        return std::ranges::equal(element_nodes, fake_nodes);
    }));
    CHECK_FALSE(MeshPartition< 1 >{}.find([](const auto&) { return true; }));
    CHECK_FALSE(MeshPartition< 1 >{}.find(0));
}

TEST_CASE("Element lookup by ID", "[mesh]")
{
    auto       domain       = Domain< 1 >{};
    const auto push_elem_id = [&](el_id_t id) {
        constexpr auto data  = ElementData< ElementType::Line, 1 >{};
        constexpr auto nodes = std::array< n_id_t, 2 >{};
        emplaceInDomain< ElementType::Line, 1 >(domain, nodes, data, id);
    };
    const auto check_contains = [&](el_id_t id) {
        auto part = MeshPartition< 1 >{MeshPartition< 1 >::domain_map_t{{0, domain}}, {}};
        CHECK(part.find(id));
        CHECK(std::as_const(part).find(id));
    };
    const auto check_doesnt_contain = [&](el_id_t id) {
        auto part = MeshPartition< 1 >{MeshPartition< 1 >::domain_map_t{{0, domain}}, {}};
        CHECK_FALSE(part.find(id));
        CHECK_FALSE(std::as_const(part).find(id));
    };

    check_doesnt_contain(0);

    el_id_t id = 1;
    push_elem_id(id++);
    check_contains(1);
    check_doesnt_contain(0);

    for (size_t i = 0; i < 10; ++i)
        push_elem_id(id++);
    check_contains(10);
    check_doesnt_contain(0);

    push_elem_id(0);
    domain.elements.visitVectors([](auto& v) { std::ranges::sort(v, {}, [](const auto& el) { return el.id; }); });
    check_contains(0);

    push_elem_id(id *= 2);
    check_contains(id);
    check_doesnt_contain(id + 1);
}

TEST_CASE("Serial mesh partitioning", "[mesh]")
{
    constexpr auto boundary_ids = std::array< d_id_t, 4 >{2, 3, 4, 5};
    constexpr auto n_parts      = 2;
    const auto     mesh         = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), boundary_ids, gmsh_tag);
    const auto     n_elements   = mesh.getNElements();
    const auto     n_nodes      = mesh.getNNodes();

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
    CHECK(partitions.front().getNodeOwnership().owned().size() + partitions.back().getNodeOwnership().owned().size() ==
          n_nodes);
    std::vector< n_id_t > intersects;
    std::ranges::set_intersection(partitions.front().getNodeOwnership().owned(),
                                  partitions.back().getNodeOwnership().owned(),
                                  std::back_inserter(intersects));
    CHECK(intersects.empty());
    intersects.clear();
    std::ranges::set_intersection(partitions.front().getNodeOwnership().shared(),
                                  partitions.back().getNodeOwnership().shared(),
                                  std::back_inserter(intersects));
    CHECK(intersects.empty());

    // Boundaries reconstructed correctly
    constexpr auto get_bnd_size = [](const MeshPartition< 1 >& part, d_id_t bnd_id) -> size_t {
        try
        {
            return part.getBoundary(bnd_id).element_views.size();
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
        CHECK(mesh.getNodeOwnership().owned().size() == 44745u);
        CHECK(std::ranges::adjacent_find(mesh.getNodeOwnership().owned(), [](auto n1, auto n2) {
                  return n2 - n1 != 1;
              }) == mesh.getNodeOwnership().owned().end());
    }

    SECTION("procedurally generated mesh")
    {
        constexpr el_o_t     order = 2;
        constexpr std::array dist{0., .2, .4, .6, .8, 10.};
        constexpr auto       n_edge_els = dist.size() - 1;
        auto                 mesh1      = makeCubeMesh(dist);
        const auto           n_elements = mesh1.getNElements();
        const auto           n_nodes_o1 = mesh1.getNodeOwnership().owned().size();
        REQUIRE(n_elements == n_edge_els * n_edge_els * n_edge_els + 6 * n_edge_els * n_edge_els);
        REQUIRE(n_nodes_o1 == dist.size() * dist.size() * dist.size());

        auto mesh = convertMeshToOrder< order >(mesh1);
        CHECK(mesh.getNElements() == n_elements);
        const auto expected_edge_nodes = (dist.size() - 1) * order + 1;
        const auto expected_n_nodes    = expected_edge_nodes * expected_edge_nodes * expected_edge_nodes;
        CHECK(mesh.getNodeOwnership().owned().size() == expected_n_nodes);
        for (size_t i = 0; i < mesh.getNodeOwnership().owned().size() - 1; ++i)
            CHECK(mesh.getNodeOwnership().owned()[i] + 1 == mesh.getNodeOwnership().owned()[i + 1]);
    }
}

template < ElementType ET, el_o_t EO, el_o_t... orders >
auto getGlobalNodes(const LocalElementView< ET, EO >& el, const MeshPartition< orders... >& mesh)
{
    auto retval = std::array< n_id_t, Element< ET, EO >::n_nodes >{};
    std::ranges::transform(
        el.getLocalNodes(), retval.begin(), [&](n_loc_id_t n) { return mesh.getNodeOwnership().getGlobalIndex(n); });
    return retval;
}

template < el_o_t O >
using Order = std::integral_constant< el_o_t, O >;
TEMPLATE_TEST_CASE("Local mesh view", "[mesh]", Order< 1 >, Order< 2 >, Order< 4 >)
{
    constexpr auto order     = TestType::value;
    constexpr auto node_dist = std::array{0., 1., 2., 3., 4.};

    constexpr auto check_local_view = [](const auto& global_mesh) {
        auto             nodes_reordered = computeNodeOrder(global_mesh);
        const auto       local_view      = LocalMeshView{global_mesh, global_mesh};
        constexpr d_id_t domain_id       = 0;
        const auto       global_domain   = global_mesh.getDomain(domain_id);
        const auto       local_domain    = local_view.getDomains().at(domain_id);

        const auto find_local_element = [&](const auto& global_element) {
            const auto found_el = local_domain.elements.find([&](const auto& local_element) {
                using local_t  = std::remove_cvref_t< decltype(local_element) >;
                using global_t = std::remove_cvref_t< decltype(global_element) >;
                if constexpr (local_t::type == global_t::type and local_t::order == global_t::order)
                {
                    const auto nodes = getGlobalNodes(local_element, global_mesh);
                    return nodes == global_element.nodes and local_element.getData() == global_element.data;
                }
                return false;
            });
            return found_el.has_value();
        };
        const auto check_element = [&](const auto& global_element) {
            CHECK(find_local_element(global_element));
        };
        global_domain.elements.visit(check_element, std::execution::seq);

        const auto find_local_boundary_view = [&](const auto& global_boundary_view, d_id_t boundary_id) {
            const auto found_el = local_domain.elements.find([&](const auto& local_element) {
                using local_t  = std::remove_cvref_t< decltype(local_element) >;
                using global_t = std::remove_cvref_t< decltype(global_boundary_view) >;
                if constexpr (local_t::type == global_t::type and local_t::order == global_t::order)
                {
                    for (const auto& [side, boundary] : local_element.getBoundaries())
                        if (boundary == boundary_id)
                        {
                            const auto nodes = getGlobalNodes(local_element, global_mesh);
                            if (nodes == global_boundary_view->nodes and
                                local_element.getData() == global_boundary_view->data and
                                side == global_boundary_view.getSide())
                                return true;
                        }
                }
                return false;
            });
            return found_el.has_value();
        };
        for (d_id_t boundary_id : global_mesh.getBoundaryIdsView())
        {
            const auto& boundary_view       = global_mesh.getBoundary(boundary_id);
            const auto  check_boundary_view = [&](const auto& global_element_boundary_view) {
                CHECK(find_local_boundary_view(global_element_boundary_view, boundary_id));
            };
            boundary_view.element_views.visit(check_boundary_view, std::execution::seq);
        }
    };

    SECTION("2D")
    {
        const auto mesh_o1 = makeSquareMesh(node_dist);
        const auto mesh    = convertMeshToOrder< order >(mesh_o1);
        check_local_view(mesh);
    }
    SECTION("3D")
    {
        const auto mesh_o1 = makeCubeMesh(node_dist);
        const auto mesh    = convertMeshToOrder< order >(mesh_o1);
        check_local_view(mesh);
    }
}

TEST_CASE("Serialize/Deserialize mesh", "[mesh]")
{
    constexpr el_o_t order        = 4;
    const auto       mesh         = convertMeshToOrder< order >(makeSquareMesh(std::array{0., 1., 2., 3.}));
    const auto       serial       = serializeMesh(mesh);
    const auto       deserialized = deserializeMesh< order >(serial);

    const auto contains_elem = [&]< ElementType T, el_o_t O >(d_id_t dom, const Element< T, O >& element) {
        const auto predicate = [&]< ElementType ET, el_o_t EO >(const Element< ET, EO >& other_element) {
            if constexpr (ET == T and EO == O)
                return element == other_element;
            else
                return false;
        };
        return deserialized.find(predicate, {dom}).has_value();
    };

    REQUIRE(mesh.getNElements() == deserialized.getNElements());
    REQUIRE(std::ranges::equal(mesh.getBoundaryIdsView(), deserialized.getBoundaryIdsView()));
    REQUIRE(std::ranges::equal(mesh.getNodeOwnership().owned(), deserialized.getNodeOwnership().owned()));
    REQUIRE(std::ranges::equal(mesh.getNodeOwnership().shared(), deserialized.getNodeOwnership().shared()));
    for (auto dom : mesh.getDomainIds())
        mesh.visit([&](const auto& element) { CHECK(contains_elem(dom, element)); }, dom);
}

TEST_CASE("Merge meshes", "[mesh]")
{
    const auto square1 = makeSquareMesh(std::array{0., 1., 2.});
    SECTION("Fully overlapping boundaries")
    {
        const auto square2 = makeSquareMesh(std::array{2., 3., 4.}, std::array{0., 1., 2.});
        const auto merged  = merge(square1, square2);

        CHECK(std::ranges::equal(merged.getDomainIds(), std::views::iota(0, 5)));
        CHECK(std::ranges::equal(merged.getBoundaryIdsView(), std::views::iota(1, 5)));
        CHECK(merged.getDomain(0).numElements() == 8);
        CHECK(merged.getDomain(1).numElements() == 4);
        CHECK(merged.getDomain(2).numElements() == 4);
        CHECK(merged.getDomain(3).numElements() == 2);
        CHECK(merged.getDomain(4).numElements() == 2);
    }
    SECTION("Partially overlapping boundaries")
    {
        const auto square2 = makeSquareMesh(
            std::array{2., 3., 4.}, std::array{1., 2., 3.}, {.bottom = 5, .top = 6, .left = 7, .right = 8});
        const auto merged = merge(square1, square2);

        CHECK(std::ranges::equal(merged.getDomainIds(), std::views::iota(0, 9)));
        CHECK(std::ranges::equal(merged.getBoundaryIdsView(), std::views::iota(1, 9)));
        CHECK(merged.getDomain(0).numElements() == 8);
        CHECK(merged.getDomain(1).numElements() == 2);
        CHECK(merged.getDomain(2).numElements() == 2);
        CHECK(merged.getDomain(3).numElements() == 2);
        CHECK(merged.getDomain(4).numElements() == 1);
        CHECK(merged.getDomain(5).numElements() == 2);
        CHECK(merged.getDomain(6).numElements() == 2);
        CHECK(merged.getDomain(7).numElements() == 1);
        CHECK(merged.getDomain(8).numElements() == 2);
    }
}
