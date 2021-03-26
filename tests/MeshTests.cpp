#include "l3ster.hpp"

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

TEMPLATE_TEST_CASE("Quadrilateral mesh", "[mesh]", lstr::Mesh, const lstr::Mesh)
{
    // Flag to prevent non-const member functions from being tested on const object
    constexpr bool is_const = std::is_same_v< TestType, const lstr::Mesh >;

    TestType mesh = lstr::readMesh(L3STER_GENERATE_ABS_TEST_DATA_PATH(gmesh_ascii4.msh), lstr::gmsh_tag);

    REQUIRE(mesh.getPartitions().size() == 1);

    auto&       topology = mesh.getPartitions()[0];
    const auto& nodes    = mesh.getNodes();

    REQUIRE(nodes.size() == 121);

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
    const auto element1  = topology.findElement(predicate);

    CHECK(element1);
    CHECK(ConstructionTracker::defaults == 1);
    CHECK(ConstructionTracker::copies == 0);
    CHECK(ConstructionTracker::cp_asgn == 0);
    CHECK(ConstructionTracker::moves == 0);
    CHECK(ConstructionTracker::mv_asgn == 0);

    ConstructionTracker::resetCount();

    // Check that find element fails safely
    const auto predicate2 = ElementFinder(std::vector< lstr::n_id_t >({153, 213, 821, 372}));
    const auto element2   = topology.findElement(predicate2);

    CHECK_FALSE(element2);

    ConstructionTracker::resetCount();

    // BoundaryView
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

TEST_CASE("Unsupported mesh formats, mesh I/O error handling", "[mesh]")
{
    lstr::Mesh mesh;
    REQUIRE_THROWS(mesh = lstr::readMesh(L3STER_GENERATE_ABS_TEST_DATA_PATH(gmesh_ascii2.msh), lstr::gmsh_tag));
    REQUIRE_THROWS(mesh = lstr::readMesh(L3STER_GENERATE_ABS_TEST_DATA_PATH(gmesh_bin2.msh), lstr::gmsh_tag));
    REQUIRE_THROWS(mesh = lstr::readMesh(L3STER_GENERATE_ABS_TEST_DATA_PATH(gmesh_bin4.msh), lstr::gmsh_tag));
    REQUIRE_THROWS(mesh = lstr::readMesh(L3STER_GENERATE_ABS_TEST_DATA_PATH(nonexistent.msh), lstr::gmsh_tag));
    REQUIRE_THROWS(
        mesh = lstr::readMesh(L3STER_GENERATE_ABS_TEST_DATA_PATH(gmsh_triangle_mesh_ascii4.msh), lstr::gmsh_tag));
}

TEST_CASE("Serial mesh partitioning", "[mesh]")
{
    constexpr auto n_parts = 2u;
    auto           mesh    = lstr::readMesh(L3STER_GENERATE_ABS_TEST_DATA_PATH(gmesh_ascii4.msh), lstr::gmsh_tag);
    lstr::partitionMesh(mesh, n_parts, {});
    REQUIRE(mesh.getPartitions().size() == n_parts);
    CHECK(mesh.getPartitions()[0].getNElements() == Approx(mesh.getPartitions()[1].getNElements()).epsilon(.1));
}
