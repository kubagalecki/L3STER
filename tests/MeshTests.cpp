#include "catch2/catch.hpp"
#include "l3ster.hpp"

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
    template < lstr::mesh::ElementTypes ET, lstr::types::el_o_t EO >
    void operator()(const lstr::mesh::Element< ET, EO >&)
    {
        ++counter;
    }

    size_t counter = 0;
};

struct ElementFinder : ConstructionTracker
{
    using nv_t = std::vector< lstr::types::n_id_t >;

    ElementFinder()                         = default;
    ElementFinder(const ElementFinder&)     = default;
    ElementFinder(ElementFinder&&) noexcept = default;
    ElementFinder& operator=(const ElementFinder&) = default;
    ElementFinder& operator=(ElementFinder&&) noexcept = default;
    explicit ElementFinder(nv_t&& nodes_) : ConstructionTracker{}, nodes{std::move(nodes_)} {}

    template < lstr::mesh::ElementTypes ET, lstr::types::el_o_t EO >
    bool operator()(const lstr::mesh::Element< ET, EO >& element) const
    {
        const auto& element_nodes = element.getNodes();

        const bool ret_val =
            std::all_of(nodes.cbegin(), nodes.cend(), [&element_nodes](lstr::types::n_id_t n) {
                return std::any_of(element_nodes.cbegin(),
                                   element_nodes.cend(),
                                   [&n](lstr::types::n_id_t en) { return en = n; });
            });

        return ret_val;
    }

    nv_t nodes;
};

TEST_CASE("Quadrilateral mesh", "[mesh]")
{
    auto mesh = lstr::mesh::readMesh("mesh_ascii4.msh", lstr::mesh::gmsh_tag);

    REQUIRE(mesh.getPartitions().size() == 1);

    auto&       topology = mesh.getPartitions()[0];
    const auto& nodes    = mesh.getNodes();

    REQUIRE(nodes.size() == 121);

    auto element_counter = topology.visitAllElements(ElementCounter{});

    REQUIRE(element_counter.counter == 140);
    CHECK(ConstructionTracker::defaults == 1);
    CHECK(ConstructionTracker::copies == 0);
    CHECK(ConstructionTracker::cp_asgn == 0);
    CHECK(ConstructionTracker::moves <= 1);
    CHECK(ConstructionTracker::mv_asgn <= 1);

    ConstructionTracker::resetCount();
    element_counter.counter = 0;

    element_counter =
        topology.visitDomainIf(std::move(element_counter),
                               [](const lstr::mesh::DomainView& dv) { return dv.getDim() == 2; });

    CHECK(element_counter.counter == 100);
    CHECK(ConstructionTracker::defaults == 0);
    CHECK(ConstructionTracker::copies == 0);
    CHECK(ConstructionTracker::cp_asgn == 0);
    CHECK(ConstructionTracker::moves <= 1);
    CHECK(ConstructionTracker::mv_asgn <= 1);

    ConstructionTracker::resetCount();
    element_counter.counter = 0;

    element_counter = topology.cvisitAllElements(std::move(element_counter));

    REQUIRE(element_counter.counter == 140);
    CHECK(ConstructionTracker::defaults == 0);
    CHECK(ConstructionTracker::copies == 0);
    CHECK(ConstructionTracker::cp_asgn == 0);
    CHECK(ConstructionTracker::moves <= 1);
    CHECK(ConstructionTracker::mv_asgn <= 1);

    ConstructionTracker::resetCount();
    element_counter.counter = 0;

    element_counter =
        topology.cvisitDomainIf(std::move(element_counter),
                                [](const lstr::mesh::DomainView& dv) { return dv.getDim() == 2; });

    CHECK(element_counter.counter == 100);
    CHECK(ConstructionTracker::defaults == 0);
    CHECK(ConstructionTracker::copies == 0);
    CHECK(ConstructionTracker::cp_asgn == 0);
    CHECK(ConstructionTracker::moves <= 1);
    CHECK(ConstructionTracker::mv_asgn <= 1);

    ConstructionTracker::resetCount();
    element_counter.counter = 0;

    const auto predicate = ElementFinder(std::vector< lstr::types::n_id_t >({54, 55, 64, 62}));
    const auto element1  = topology.findElement(predicate);

    CHECK(element1.has_value() == true);
    CHECK(ConstructionTracker::defaults == 1);
    CHECK(ConstructionTracker::copies == 0);
    CHECK(ConstructionTracker::cp_asgn == 0);
    CHECK(ConstructionTracker::moves == 0);
    CHECK(ConstructionTracker::mv_asgn == 0);

    ConstructionTracker::resetCount();
}
