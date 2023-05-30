#include "l3ster/assembly/MakeTpetraMap.hpp"
#include "l3ster/assembly/NodeToDofMap.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"

#include "catch2/catch.hpp"

#include "TestDataPath.h"

using namespace lstr;

template < el_o_t... orders, CondensationPolicy CP, detail::ProblemDef_c auto problem_def >
static auto makeLocalCondensationMap(const MeshPartition< orders... >& mesh,
                                     ConstexprValue< problem_def >     probdef_ctwrpr,
                                     CondensationPolicyTag< CP > = {}) -> detail::NodeCondensationMap< CP >
{
    const auto active_nodes    = detail::getActiveNodes< CP >(mesh, probdef_ctwrpr);
    auto       condensed_nodes = std::vector< n_id_t >(active_nodes);
    std::iota(condensed_nodes.begin(), condensed_nodes.end(), 0);
    return {active_nodes, std::move(condensed_nodes)};
}

TEMPLATE_TEST_CASE("Local node DOF interval calculation",
                   "[dof]",
                   CondensationPolicyTag< CondensationPolicy::None >,
                   CondensationPolicyTag< CondensationPolicy::ElementBoundary >)
{
    constexpr auto   node_dist      = std::array{0., 1., 2.};
    const auto       mesh           = makeCubeMesh(node_dist);
    constexpr size_t n_fields       = 2;
    constexpr auto   probdef_ctwrpr = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{true, false}},
                                                               Pair{d_id_t{1}, std::array{false, true}}} >{};
    const auto       cond_map       = makeLocalCondensationMap(mesh, probdef_ctwrpr, TestType{});
    const auto       result         = detail::computeLocalDofIntervals(mesh, cond_map, probdef_ctwrpr);
    REQUIRE(result.size() == 2);
    CHECK(result[0].first[0] == 0);
    CHECK(result[0].first[1] == 8);
    CHECK(result[0].second == std::bitset< n_fields >{0b11});
    CHECK(result[1].first[0] == 9);
    CHECK(result[1].first[1] == 26);
    CHECK(result[1].second == std::bitset< n_fields >{0b01});
}

TEMPLATE_TEST_CASE("Node DOF intervals of parts sum up to whole",
                   "[dof]",
                   CondensationPolicyTag< CondensationPolicy::None >,
                   CondensationPolicyTag< CondensationPolicy::ElementBoundary >)
{
    constexpr auto problem_def   = ConstexprValue< std::array{Pair{d_id_t{1}, std::array{false, false, false}},
                                                            Pair{d_id_t{2}, std::array{false, false, true}},
                                                            Pair{d_id_t{3}, std::array{false, true, false}},
                                                            Pair{d_id_t{4}, std::array{false, true, true}},
                                                            Pair{d_id_t{5}, std::array{true, false, false}},
                                                            Pair{d_id_t{6}, std::array{true, false, true}},
                                                            Pair{d_id_t{7}, std::array{true, true, false}},
                                                            Pair{d_id_t{8}, std::array{true, true, true}}} >{};
    const auto     mesh          = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube_multidom.msh), gmsh_tag);
    const auto     cond_map_full = makeLocalCondensationMap(mesh, problem_def, TestType{});
    const auto     unpartitioned_intervals = detail::computeLocalDofIntervals(mesh, cond_map_full, problem_def);

    std::vector< d_id_t > boundaries(24);
    std::iota(boundaries.begin(), boundaries.end(), 9);
    const auto partitions            = partitionMesh(mesh, 2, boundaries);
    auto       partitioned_intervals = unpartitioned_intervals;
    partitioned_intervals.clear();
    for (const auto& part : partitions)
    {
        const auto cond_map  = std::invoke([&] {
            // All nodes are boundary nodes since mesh is of the 1st order
            // Condensation map without interal nodes is the identity mapping
            auto cond_ids = std::vector< n_id_t >{};
            std::ranges::copy(part.getAllNodes(), std::back_inserter(cond_ids));
            std::ranges::sort(cond_ids);
            return detail::NodeCondensationMap< TestType::value >{cond_ids, cond_ids};
        });
        const auto intervals = detail::computeLocalDofIntervals(part, cond_map, problem_def);
        std::ranges::copy(intervals, std::back_inserter(partitioned_intervals));
    }
    detail::consolidateDofIntervals(partitioned_intervals);
    CHECK(partitioned_intervals == unpartitioned_intervals);
}

TEMPLATE_TEST_CASE("Node DOF interval (de-)serialization", "[dof]", ConstexprValue< 10 >, ConstexprValue< 100 >)
{
    constexpr std::size_t n_fields           = TestType::value;
    constexpr size_t      test_size          = 1u << 12;
    auto                  original_intervals = detail::node_interval_vector_t< n_fields >{};
    original_intervals.reserve(test_size);
    std::generate_n(
        std::back_inserter(original_intervals), test_size, [prng = std::mt19937{std::random_device{}()}]() mutable {
            std::uniform_int_distribution< n_id_t >             delim_dist{};
            std::uniform_int_distribution< unsigned long long > field_dist{};
            return std::make_pair(std::array< n_id_t, 2 >{delim_dist(prng), delim_dist(prng)},
                                  std::bitset< n_fields >{field_dist(prng)});
        });

    auto deserialized_intervals = original_intervals;
    deserialized_intervals.clear();
    std::vector< unsigned long long > serial;
    serial.reserve((util::bitsetNUllongs< n_fields >() + 2) * original_intervals.size());
    detail::serializeDofIntervals(original_intervals, std::back_inserter(serial));
    detail::deserializeDofIntervals< n_fields >(serial, std::back_inserter(deserialized_intervals));
    CHECK(original_intervals == deserialized_intervals);
}

TEMPLATE_TEST_CASE("Node DOF interval consolidation", "[dof]", ConstexprValue< 10 >, ConstexprValue< 100 >)
{
    std::mt19937 prng{std::random_device{}()};
    int          n_runs          = 1000; // This is effectively a fuzz test, so n_runs needs to be sufficiently large
    int          min_n_intervals = 0, max_n_intervals = 1 << 8;
    n_id_t       min_delim = 0, max_delim = 1u << 12;

    constexpr size_t n_fields        = TestType::value;
    using delim_t                    = std::array< n_id_t, 2 >;
    using cov_t                      = std::bitset< n_fields >;
    using interval_t                 = std::pair< delim_t, cov_t >;
    const auto& make_random_interval = [&](n_id_t d_min, n_id_t d_max) {
        delim_t delims;
        delims[0] = std::uniform_int_distribution< n_id_t >{d_min, d_max}(prng);
        delims[1] = std::uniform_int_distribution< n_id_t >{d_min, d_max}(prng);
        std::ranges::sort(delims);
        std::array< unsigned long long, util::bitsetNUllongs< n_fields >() > cov_serial{};
        std::generate_n(begin(cov_serial), cov_serial.size(), [&] {
            return std::uniform_int_distribution< unsigned long long >{}(prng);
        });
        const auto cov = util::trimBitset< n_fields >(util::deserializeBitset(cov_serial));
        return std::make_pair(delims, cov);
    };

    for (int run = 0; run < n_runs; ++run)
    {
        const auto n_intervals = std::uniform_int_distribution< int >{min_n_intervals, max_n_intervals}(prng);
        const auto n_nodes     = std::uniform_int_distribution< n_id_t >{min_delim + 1, max_delim + 1}(prng);

        auto                      node_covs = std::vector< cov_t >(n_nodes);
        std::vector< interval_t > intervals;
        intervals.reserve(n_intervals);
        std::generate_n(std::back_inserter(intervals), n_intervals, [&] {
            const auto interval = make_random_interval(0, n_nodes - 1);
            for (const auto [lo, hi] = interval.first; auto node : std::views::iota(lo, hi + 1))
                node_covs[node] |= interval.second;
            return interval;
        });

        detail::consolidateDofIntervals(intervals);

        if (intervals.empty())
        {
            const bool empty_intervals_handled_correctly = n_intervals == 0;
            CHECK(empty_intervals_handled_correctly);
            continue;
        }

        const bool intervals_are_sorted =
            std::ranges::is_sorted(intervals, [](const auto& i1, const auto& i2) { return i1.first < i2.first; });
        CHECK(intervals_are_sorted);
        const bool intervals_are_disjoint =
            std::ranges::all_of(intervals | std::views::drop(1) | std::views::keys,
                                [prev = intervals[0].first[1]](const delim_t& delim) mutable {
                                    const auto [lo, hi] = delim;
                                    return lo > std::exchange(prev, hi);
                                });
        CHECK(intervals_are_disjoint);
        const bool intervals_are_correct = std::ranges::all_of(intervals, [&](const auto& interval) {
            const auto& [delims, cov] = interval;
            const auto [lo, hi]       = delims;
            return std::ranges::all_of(std::views::iota(lo, hi + 1), [&](auto node) { return node_covs[node] == cov; });
        });
        CHECK(intervals_are_correct);
    }
}

TEMPLATE_TEST_CASE("Node to global DOF",
                   "[dof]",
                   CondensationPolicyTag< CondensationPolicy::None >,
                   CondensationPolicyTag< CondensationPolicy::ElementBoundary >)
{
    constexpr std::array node_dist = {0., 1., 2., 3., 4.};
    constexpr size_t     n_parts   = 4;
    const auto           mesh      = makeCubeMesh(node_dist);
    constexpr size_t     n_fields  = 3;

    constexpr auto probdef_domain = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{true}}} >{};
    const auto     cond_map_full  = makeLocalCondensationMap(mesh, probdef_domain, TestType{});

    SECTION("Non-contiguous DOF distribution")
    {
        auto   dof_intervals = detail::node_interval_vector_t< n_fields >{};
        n_id_t i1_begin = 0, i1_end = node_dist.size() * node_dist.size() - 1, i2_begin = i1_end + 1,
               i2_end = mesh.getOwnedNodes().size() - 1;
        std::bitset< n_fields > i1_cov{0b110ull}, i2_cov{0b101ull};
        dof_intervals.emplace_back(std::array{i1_begin, i1_end}, i1_cov);
        dof_intervals.emplace_back(std::array{i2_begin, i2_end}, i2_cov);

        const auto check_dofs = [&](std::span< const n_id_t > nodes, const NodeToGlobalDofMap< n_fields >& map) {
            for (auto node : nodes)
            {
                const auto  computed_dofs = map(cond_map_full.getCondensedId(node));
                const auto  expected_dof_base{static_cast< global_dof_t >(
                    node > i1_end ? (i1_end - i1_begin + 1) * i1_cov.count() + (node - i2_begin) * i2_cov.count()
                                   : (node - i1_begin) * i1_cov.count())};
                const auto& cov = node > i1_end ? i2_cov : i1_cov;

                global_dof_t node_dof_count = 0;
                for (size_t local_dof_ind = 0; auto computed_dof : computed_dofs)
                    if (cov.test(local_dof_ind++))
                    {
                        const global_dof_t expected_dof = expected_dof_base + node_dof_count++;
                        CHECK(computed_dof == expected_dof);
                    }
            }
        };

        SECTION("Unpartitioned")
        {
            const auto map = NodeToGlobalDofMap{dof_intervals, cond_map_full};
            CHECK_FALSE(map.isContiguous());
            check_dofs(mesh.getOwnedNodes(), map);
        }

        SECTION("Partitioned")
        {
            const auto partitions     = partitionMesh(mesh, n_parts, {});
            bool       not_contiguous = false;
            for (const auto& part : partitions)
            {
                const auto map = NodeToGlobalDofMap{dof_intervals, cond_map_full};
                not_contiguous |= not map.isContiguous();
                check_dofs(part.getOwnedNodes(), map);
                check_dofs(part.getGhostNodes(), map);
            }
            CHECK(not_contiguous); // at least one of the partitions should have a non-contiguous map
        }
    }

    SECTION("Contiguous DOF distribution")
    {
        detail::node_interval_vector_t< n_fields > dof_intervals;
        const auto                                 max_node = node_dist.size() * node_dist.size() * node_dist.size();
        dof_intervals.emplace_back(std::array< n_id_t, 2 >{0, max_node}, std::bitset< n_fields >{0b111ul});

        const auto check_dofs = [&](std::span< const n_id_t > nodes, const NodeToGlobalDofMap< n_fields >& map) {
            for (auto node : nodes)
            {
                const auto   computed_dofs = map(node);
                global_dof_t dof           = node * n_fields;
                for (auto mapped_dof : computed_dofs)
                    CHECK(mapped_dof == dof++);
            }
        };

        SECTION("Unpartitioned")
        {
            const auto map = NodeToGlobalDofMap{dof_intervals, cond_map_full};
            CHECK(map.isContiguous());
            check_dofs(mesh.getOwnedNodes(), map);
        }

        SECTION("Partitioned")
        {
            const auto partitions = partitionMesh(mesh, n_parts, {});
            for (const auto& part : partitions)
            {
                const auto map = NodeToGlobalDofMap{dof_intervals, cond_map_full};
                CHECK(map.isContiguous());
                check_dofs(part.getOwnedNodes(), map);
                check_dofs(part.getGhostNodes(), map);
            }
        }
    }
}
