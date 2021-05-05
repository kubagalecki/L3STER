#include "l3ster.hpp"
#include "mesh/ConvertMeshToOrder.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

#include <algorithm>
#include <vector>

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
    template < lstr::ElementTypes ET, lstr::el_o_t EO >
    void operator()(const lstr::Element< ET, EO >&)
    {
        ++counter;
    }

    size_t counter = 0;
};

struct ElementFinder : ConstructionTracker
{
    using nv_t = std::vector< lstr::n_id_t >;

    explicit ElementFinder(nv_t&& nodes_) : ConstructionTracker{}, nodes{std::move(nodes_)}
    {
        std::ranges::sort(nodes);
    }

    template < lstr::ElementTypes ET, lstr::el_o_t EO >
    bool operator()(const lstr::Element< ET, EO >& element) const
    {
        auto element_nodes = element.getNodes();
        std::ranges::sort(element_nodes);
        return std::ranges::equal(element_nodes, nodes);
    }

    nv_t nodes;
};

TEMPLATE_TEST_CASE("2D mesh import", "[mesh]", lstr::Mesh, const lstr::Mesh)
{
    // Flag to prevent non-const member functions from being tested on const object
    constexpr bool is_const = std::is_same_v< TestType, const lstr::Mesh >;

    TestType mesh = lstr::readMesh(L3STER_TESTDATA_ABSPATH(gmesh_ascii4.msh), lstr::gmsh_tag);

    REQUIRE(mesh.getPartitions().size() == 1);

    auto&       topology = mesh.getPartitions()[0];
    const auto& nodes    = mesh.getVertices();

    CHECK_THROWS(topology.getDualGraph());

    REQUIRE(nodes.size() == 121);
    REQUIRE(nodes.size() == topology.getNodes().size());
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
        topology.cvisit(std::move(element_counter), [](const lstr::DomainView& dv) { return dv.getDim() == 2; });

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
            topology.visit(std::move(element_counter), [](const lstr::DomainView& dv) { return dv.getDim() == 2; });

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
            topology.cvisit(std::move(element_counter), [](const lstr::DomainView& dv) { return dv.getDim() == 2; });

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
    const auto predicate = ElementFinder(std::vector< lstr::n_id_t >({54, 55, 64, 63}));
    const auto element1  = topology.find(predicate);

    CHECK(element1);
    CHECK(ConstructionTracker::defaults == 1);
    CHECK(ConstructionTracker::copies == 0);
    CHECK(ConstructionTracker::cp_asgn == 0);
    CHECK(ConstructionTracker::moves == 0);
    CHECK(ConstructionTracker::mv_asgn == 0);

    ConstructionTracker::resetCount();

    // Check that find element fails safely
    const auto predicate2 = ElementFinder(std::vector< lstr::n_id_t >({153, 213, 821, 372}));
    const auto element2   = topology.find(predicate2);

    CHECK_FALSE(element2);

    ConstructionTracker::resetCount();

    // BoundaryView
    if constexpr (!is_const)
        topology.initDualGraph();
    std::vector< lstr::BoundaryView > boundaries;
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
    auto  mesh = lstr::readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube.msh), lstr::gmsh_tag);
    auto& part = mesh.getPartitions()[0];
    part.initDualGraph();

    constexpr size_t expected_nvertices = 5885;
    constexpr size_t expected_nelements = 5664;
    CHECK(mesh.getVertices().size() == expected_nvertices);
    CHECK(part.getNElements() == expected_nelements);

    for (int i = 2; i <= 7; ++i)
        CHECK_NOTHROW(part.getBoundaryView(i));

    CHECK_THROWS(part.getBoundaryView(42));
}

TEST_CASE("Unsupported mesh formats, mesh I/O error handling", "[mesh]")
{
    using lstr::gmsh_tag;
    using lstr::readMesh;
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmesh_ascii2.msh), gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmesh_bin2.msh), gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmesh_bin4.msh), gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(nonexistent.msh), gmsh_tag));
    CHECK_THROWS(readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_triangle_mesh.msh), gmsh_tag));
}

TEST_CASE("Element lookup by ID", "[mesh]")
{
    lstr::Domain d;
    CHECK_FALSE(d.find(1));

    std::array< lstr::n_id_t, 2 > nodes{1, 2};
    lstr::el_id_t                 id = 1;
    d.emplaceBack< lstr::ElementTypes::Line, 1 >(nodes, id++);
    CHECK_FALSE(d.find(0));
    CHECK(d.find(1));

    const auto next = [&]() -> std::array< lstr::n_id_t, 2 >& {
        std::ranges::for_each(nodes, [](auto& n) { ++n; });
        return nodes;
    };

    for (size_t i = 0; i < 10; ++i)
        d.emplaceBack< lstr::ElementTypes::Line, 1 >(next(), id++);
    CHECK(d.find(10));

    d.emplaceBack< lstr::ElementTypes::Line, 1 >(next(), 0);
    CHECK(d.find(0));

    d.emplaceBack< lstr::ElementTypes::Line, 1 >(next(), id *= 2);
    CHECK(d.find(id));
    CHECK_FALSE(d.find(id + 1));
}

TEST_CASE("Serial mesh partitioning", "[mesh]")
{
    constexpr auto n_parts    = 2u;
    auto           mesh       = lstr::readMesh(L3STER_TESTDATA_ABSPATH(gmesh_ascii4.msh), lstr::gmsh_tag);
    const auto     n_elements = mesh.getPartitions()[0].getNElements();
    REQUIRE_NOTHROW(lstr::partitionMesh(mesh, 1, {}));
    lstr::partitionMesh(mesh, n_parts, {2, 3, 4, 5});
    REQUIRE(mesh.getPartitions().size() == n_parts);
    REQUIRE(std::accumulate(
                mesh.getPartitions().cbegin(), mesh.getPartitions().cend(), 0u, [](size_t size, const auto& part) {
                    return size + part.getNElements();
                }) == n_elements);
    auto& p1 = mesh.getPartitions()[0];
    auto& p2 = mesh.getPartitions()[1];
    CHECK(p1.getNElements() == Approx(p2.getNElements()).epsilon(.1));
    std::vector< lstr::n_id_t > intersects;
    std::ranges::set_intersection(p1.getNodes(), p2.getNodes(), std::back_inserter(intersects));
    CHECK(intersects.empty());
    intersects.clear();
    std::ranges::set_intersection(p1.getGhostNodes(), p2.getGhostNodes(), std::back_inserter(intersects));
    CHECK(intersects.empty());
    REQUIRE_THROWS(lstr::partitionMesh(mesh, 42, {}));
}

TEST_CASE("Mesh conversion to higher order", "[mesh]")
{
    constexpr lstr::el_o_t order      = 2;
    auto                   mesh       = lstr::readMesh(L3STER_TESTDATA_ABSPATH(gmesh_ascii4.msh), lstr::gmsh_tag);
    auto&                  part       = mesh.getPartitions()[0];
    const auto             n_elements = part.getNElements();
    lstr::convertMeshToOrder< order >(mesh);
    CHECK(n_elements == part.getNElements());
    const auto validate_elorder = [&]< lstr::ElementTypes T, lstr::el_o_t O >(const lstr::Element< T, O >&) {
        if constexpr (O != order)
            throw std::logic_error{"Incorrect element order"};
    };
    CHECK_NOTHROW(part.cvisit(validate_elorder));
}
