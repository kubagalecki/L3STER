#include "l3ster/assembly/SolutionManager.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"

#include "catch2/catch.hpp"

using namespace lstr;

TEST_CASE("Solution Manager", "[sol_man]")
{
    constexpr auto   node_dist  = std::array{0., 1., 2., 3., 4., 5.};
    constexpr size_t n_parts    = 4;
    const auto       mesh       = mesh::makeCubeMesh(node_dist);
    const auto       partitions = mesh::partitionMesh(mesh, n_parts, {});
    constexpr size_t n_fields   = 3;
    auto             sol_mans   = std::vector< SolutionManager >{};
    std::ranges::transform(
        partitions, std::back_inserter(sol_mans), [&, i = 0](const mesh::MeshPartition< 1 >& part) mutable {
            return SolutionManager{part, n_fields, static_cast< val_t >(i++)};
        });

    REQUIRE(std::ranges::all_of(sol_mans, [&](const SolutionManager& sm) { return sm.nFields() == n_fields; }));
    for (size_t i = 0; const auto& sm : sol_mans)
        REQUIRE(sm.nNodes() == partitions.at(i++).getAllNodes().size());

    for (auto& sm : sol_mans)
        for (size_t i = 0; i != n_fields; ++i)
        {
            CHECK(sm.getFieldView(i).data() == std::as_const(sm).getFieldView(i).data());
            CHECK(sm.getFieldView(i).size() == std::as_const(sm).getFieldView(i).size());
        }

    constexpr auto field_inds = util::makeIotaArray< size_t, n_fields >();
    for (size_t i = 0; const auto& sm : sol_mans)
    {
        const auto& part = partitions.at(i);
        part.visit(
            [&](const auto& element) {
                for (auto n : element.getNodes())
                    for (auto fv : sm.getNodeValues(n, std::span{field_inds}))
                        CHECK(fv == static_cast< val_t >(i));
            },
            std::execution::par);
        ++i;
    }

    for (auto& sm : sol_mans)
        for (auto i : field_inds)
            sm.setField(i, 42.);
    for (size_t i = 0; const auto& sm : sol_mans)
    {
        const auto& part = partitions.at(i);
        part.visit(
            [&](const auto& element) {
                for (auto n : element.getNodes())
                    for (auto fv : sm.getNodeValues(n, std::span{field_inds}))
                        CHECK(fv == 42.);
            },
            std::execution::par);
        ++i;
    }
}