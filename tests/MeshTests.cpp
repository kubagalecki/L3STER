#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

#include <algorithm>
#include <vector>

using namespace lstr;

struct ConstructionTracker
{
    ConstructionTracker() { ++defaults; }
    ConstructionTracker(const ConstructionTracker&) { ++copies; }
    ConstructionTracker(ConstructionTracker&&) noexcept { ++moves; }
    ConstructionTracker& operator=(const ConstructionTracker&)
    {
        ++cp_asgn;
        return *this;
    }
    ConstructionTracker& operator=(ConstructionTracker&&) noexcept
    {
        ++mv_asgn;
        return *this;
    }
    ~ConstructionTracker() = default;

    static size_t defaults;
    static size_t copies;
    static size_t moves;
    static size_t cp_asgn;
    static size_t mv_asgn;

    static void resetCount()
    {
        defaults = 0;
        copies   = 0;
        moves    = 0;
        cp_asgn  = 0;
        mv_asgn  = 0;
    }
};

size_t ConstructionTracker::defaults = 0;
size_t ConstructionTracker::copies   = 0;
size_t ConstructionTracker::moves    = 0;
size_t ConstructionTracker::cp_asgn  = 0;
size_t ConstructionTracker::mv_asgn  = 0;

struct ElementCounter : ConstructionTracker
{
    template < ElementTypes ET, el_o_t EO >
    void operator()(const Element< ET, EO >&)
    {
        ++counter;
    }

    size_t counter = 0;
};

struct ElementFinder : ConstructionTracker
{
    using nv_t = std::vector< n_id_t >;

    explicit ElementFinder(nv_t&& nodes_) : ConstructionTracker{}, nodes{std::move(nodes_)}
    {
        std::ranges::sort(nodes);
    }

    template < ElementTypes ET, el_o_t EO >
    bool operator()(const Element< ET, EO >& element) const
    {
        auto element_nodes = element.getNodes();
        std::ranges::sort(element_nodes);
        return std::ranges::equal(element_nodes, nodes);
    }

    nv_t nodes;
};

TEMPLATE_TEST_CASE("2D mesh import", "[mesh]", Mesh, const Mesh)
{
    // Flag to prevent non-const member functions from being tested on const object
    constexpr bool is_const = std::is_same_v< TestType, const Mesh >;

    TestType mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), gmsh_tag);

    REQUIRE(mesh.getPartitions().size() == 1);

    auto&            topology         = mesh.getPartitions()[0];
    constexpr size_t expected_n_nodes = 121;
    size_t           max_node{};
    const auto       update_max_node = [&](const auto& el) {
        const size_t max_el_node = *std::ranges::max_element(el.getNodes());
        if (max_el_node >= expected_n_nodes)
            throw std::invalid_argument{"max nodes exceeded"};
        max_node = std::max(max_node, max_el_node);
    };
    topology.cvisit(update_max_node);

    CHECK_THROWS(topology.getDualGraph());

    REQUIRE(max_node + 1 == expected_n_nodes);
    REQUIRE(max_node + 1 == topology.getNodes().size());
    REQUIRE(topology.getGhostNodes().size() == 0);

    // Const visitors
    auto element_counter = topology.cvisit(ElementCounter{});

    REQUIRE(element_counter.counter == 140);
    CHECK(ConstructionTracker::defaults == 1);
    CHECK(ConstructionTracker::copies == 0);
    CHECK(ConstructionTracker::cp_asgn == 0);
    CHECK(ConstructionTracker::moves <= 1);
    CHECK(ConstructionTracker::mv_asgn <= 1);

    ConstructionTracker::resetCount();
    element_counter.counter = 0;

    element_counter =
        topology.cvisit(std::move(element_counter), [](const DomainView& dv) { return dv.getDim() == 2; });

    CHECK(element_counter.counter == 100);
    CHECK(ConstructionTracker::defaults == 0);
    CHECK(ConstructionTracker::copies == 0);
    CHECK(ConstructionTracker::cp_asgn == 0);
    CHECK(ConstructionTracker::moves <= 1);
    CHECK(ConstructionTracker::mv_asgn <= 1);

    ConstructionTracker::resetCount();
    element_counter.counter = 0;

    // Non-const visitors
    if constexpr (!is_const)
    {
        element_counter = topology.visit(std::move(element_counter));

        REQUIRE(element_counter.counter == 140);
        CHECK(ConstructionTracker::defaults == 0);
        CHECK(ConstructionTracker::copies == 0);
        CHECK(ConstructionTracker::cp_asgn == 0);
        CHECK(ConstructionTracker::moves <= 1);
        CHECK(ConstructionTracker::mv_asgn <= 1);

        ConstructionTracker::resetCount();
        element_counter.counter = 0;

        element_counter =
            topology.visit(std::move(element_counter), [](const DomainView& dv) { return dv.getDim() == 2; });

        CHECK(element_counter.counter == 100);
        CHECK(ConstructionTracker::defaults == 0);
        CHECK(ConstructionTracker::copies == 0);
        CHECK(ConstructionTracker::cp_asgn == 0);
        CHECK(ConstructionTracker::moves <= 1);
        CHECK(ConstructionTracker::mv_asgn <= 1);

        ConstructionTracker::resetCount();
        element_counter.counter = 0;
    }
    else
    {
        element_counter = topology.cvisit(std::move(element_counter));

        REQUIRE(element_counter.counter == 140);
        CHECK(ConstructionTracker::defaults == 0);
        CHECK(ConstructionTracker::copies == 0);
        CHECK(ConstructionTracker::cp_asgn == 0);
        CHECK(ConstructionTracker::moves <= 1);
        CHECK(ConstructionTracker::mv_asgn <= 1);

        ConstructionTracker::resetCount();
        element_counter.counter = 0;

        element_counter =
            topology.cvisit(std::move(element_counter), [](const DomainView& dv) { return dv.getDim() == 2; });

        CHECK(element_counter.counter == 100);
        CHECK(ConstructionTracker::defaults == 0);
        CHECK(ConstructionTracker::copies == 0);
        CHECK(ConstructionTracker::cp_asgn == 0);
        CHECK(ConstructionTracker::moves <= 1);
        CHECK(ConstructionTracker::mv_asgn <= 1);

        ConstructionTracker::resetCount();
        element_counter.counter = 0;
    }

    // Find (both const and non-const)
    const auto predicate = ElementFinder(std::vector< n_id_t >({54, 55, 64, 63}));
    const auto element1  = topology.find(predicate);

    CHECK(element1);
    CHECK(ConstructionTracker::defaults == 1);
    CHECK(ConstructionTracker::copies == 0);
    CHECK(ConstructionTracker::cp_asgn == 0);
    CHECK(ConstructionTracker::moves == 0);
    CHECK(ConstructionTracker::mv_asgn == 0);

    ConstructionTracker::resetCount();

    // Check that find element fails safely
    const auto predicate2 = ElementFinder(std::vector< n_id_t >({153, 213, 821, 372}));
    const auto element2   = topology.find(predicate2);

    CHECK_FALSE(element2);

    ConstructionTracker::resetCount();

    // BoundaryView
    if constexpr (!is_const)
        topology.initDualGraph();
    std::vector< BoundaryView > boundaries;
    boundaries.reserve(4);
    for (int i = 2; i <= 5; ++i)
        boundaries.push_back(mesh.getPartitions()[0].getBoundaryView(i));

    CHECK(boundaries[0].size() == 10);
    CHECK(boundaries[1].size() == 10);
    CHECK(boundaries[2].size() == 10);
    CHECK(boundaries[3].size() == 10);

    CHECK_THROWS(mesh.getPartitions()[0].getBoundaryView(6));
}

TEST_CASE("3D mesh import", "[mesh]")
{
    auto  mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube.msh), gmsh_tag);
    auto& part = mesh.getPartitions()[0];
    part.initDualGraph();

    //    constexpr size_t expected_nvertices = 5885;
    constexpr size_t expected_nelements = 5664;
    CHECK(part.getNElements() == expected_nelements);

    for (int i = 2; i <= 7; ++i)
        CHECK_NOTHROW(part.getBoundaryView(i));

    CHECK_THROWS(part.getBoundaryView(42));
}

TEST_CASE("Unsupported mesh formats, mesh I/O error handling", "[mesh]")
{
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii2.msh), gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_bin2.msh), gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_bin4.msh), gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(nonexistent.msh), gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_triangle_mesh.msh), gmsh_tag));
}

TEST_CASE("Element lookup by ID", "[mesh]")
{
    Domain d;
    CHECK_FALSE(d.find(1));

    ElementData< ElementTypes::Line, 1 > data{{Point{0., 0., 0.}, Point{0., 0., 0.}}};

    std::array< n_id_t, 2 > nodes{1, 2};
    el_id_t                 id = 1;
    d.emplaceBack< ElementTypes::Line, 1 >(nodes, data, id++);
    CHECK_FALSE(d.find(0));
    CHECK(d.find(1));

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

TEST_CASE("Reference to physical mapping", "[mesh]")
{
    constexpr auto el_o = 2;
    SECTION("1D")
    {
        constexpr auto el_t                = ElementTypes::Line;
        using element_type                 = Element< el_t, el_o >;
        constexpr auto            el_nodes = typename element_type::node_array_t{};
        ElementData< el_t, el_o > data{{Point{1., 1., 1.}, Point{.5, .5, .5}}};
        const auto                element = element_type{el_nodes, data, 0};
        const auto                mapped  = mapToPhysicalSpace(element, Point{0.});
        CHECK(mapped.x() == Approx(.75).epsilon(1e-15));
        CHECK(mapped.y() == Approx(.75).epsilon(1e-15));
        CHECK(mapped.z() == Approx(.75).epsilon(1e-15));
    }

    SECTION("2D")
    {
        constexpr auto el_t                = ElementTypes::Quad;
        using element_type                 = Element< el_t, el_o >;
        constexpr auto            el_nodes = typename element_type::node_array_t{};
        ElementData< el_t, el_o > data{{Point{1., -1., 0.}, Point{2., -1., 0.}, Point{1., 1., 1.}, Point{2., 1., 1.}}};
        const auto                element = element_type{el_nodes, data, 0};
        const auto                mapped  = mapToPhysicalSpace(element, Point{.5, -.5});
        CHECK(mapped.x() == Approx(1.75).epsilon(1e-15));
        CHECK(mapped.y() == Approx(-.5).epsilon(1e-15));
        CHECK(mapped.z() == Approx(.25).epsilon(1e-15));
    }

    SECTION("3D")
    {
        constexpr auto el_t                = ElementTypes::Hex;
        using element_type                 = Element< el_t, el_o >;
        constexpr auto            el_nodes = typename element_type::node_array_t{};
        ElementData< el_t, el_o > data{{Point{.5, .5, .5},
                                        Point{1., .5, .5},
                                        Point{.5, 1., .5},
                                        Point{1., 1., .5},
                                        Point{.5, .5, 1.},
                                        Point{1., .5, 1.},
                                        Point{.5, 1., 1.},
                                        Point{1., 1., 1.}}};
        const auto                element = element_type{el_nodes, data, 0};
        const auto                mapped  = mapToPhysicalSpace(element, Point{0., 0., 0.});
        CHECK(mapped.x() == Approx(.75).epsilon(1e-15));
        CHECK(mapped.y() == Approx(.75).epsilon(1e-15));
        CHECK(mapped.z() == Approx(.75).epsilon(1e-15));
    }
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
        CHECK_NOTHROW(part.cvisit(validate_elorder));
        CHECK(part.getNodes().size() == 44745u);
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
        CHECK_NOTHROW(part.cvisit(validate_elorder));
    }
}
