#include "l3ster/assembly/MakeTpetraMap.hpp"
#include "l3ster/assembly/NodeToDofMap.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"

#include "catch2/catch.hpp"

#include "TestDataPath.h"

using namespace lstr;

TEST_CASE("Local node DOF interval calculation", "[dof]")
{
    constexpr std::array  node_dist   = {0., 1., 2.};
    const auto            mesh        = makeCubeMesh(node_dist);
    const auto&           part        = mesh.getPartitions()[0];
    constexpr std::size_t n_fields    = 2;
    constexpr auto        problem_def = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{true, false}},
                                                            Pair{d_id_t{1}, std::array{false, true}}} >{};
    const auto            result      = detail::computeLocalDofIntervals(part, problem_def);
    REQUIRE(result.size() == 2);
    CHECK(result[0].first[0] == 0);
    CHECK(result[0].first[1] == 8);
    CHECK(result[0].second == std::bitset< n_fields >{0b11});
    CHECK(result[1].first[0] == 9);
    CHECK(result[1].first[1] == 26);
    CHECK(result[1].second == std::bitset< n_fields >{0b01});
}

TEST_CASE("Node DOF intervals of parts sum up to whole", "[dof]")
{
    constexpr auto problem_def = ConstexprValue< std::array{Pair{d_id_t{1}, std::array{false, false, false}},
                                                            Pair{d_id_t{2}, std::array{false, false, true}},
                                                            Pair{d_id_t{3}, std::array{false, true, false}},
                                                            Pair{d_id_t{4}, std::array{false, true, true}},
                                                            Pair{d_id_t{5}, std::array{true, false, false}},
                                                            Pair{d_id_t{6}, std::array{true, false, true}},
                                                            Pair{d_id_t{7}, std::array{true, true, false}},
                                                            Pair{d_id_t{8}, std::array{true, true, true}}} >{};
    auto           mesh        = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube_multidom.msh), gmsh_tag);
    auto&          full_part   = mesh.getPartitions()[0];
    const auto     unpartitioned_intervals = detail::computeLocalDofIntervals(full_part, problem_def);

    std::vector< d_id_t > boundaries(24);
    std::iota(boundaries.begin(), boundaries.end(), 9);
    mesh = partitionMesh(mesh, 2, boundaries);
    typename std::remove_const_t< decltype(unpartitioned_intervals) > partitioned_intervals;
    partitioned_intervals.reserve(unpartitioned_intervals.size());
    for (const auto& part : mesh.getPartitions())
    {
        const auto cur_ints = detail::computeLocalDofIntervals(part, problem_def);
        partitioned_intervals.insert(partitioned_intervals.end(), cur_ints.begin(), cur_ints.end());
    }
    detail::consolidateDofIntervals(partitioned_intervals);
    CHECK(partitioned_intervals == unpartitioned_intervals);
}

TEMPLATE_TEST_CASE("Node DOF interval (de-)serialization", "[dof]", ConstexprValue< 10 >, ConstexprValue< 100 >)
{
    constexpr std::size_t n_fields = TestType::value;
    constexpr auto        dummy_problem_def =
        ConstexprValue< std::array< Pair< d_id_t, std::array< bool, n_fields > >, 1 >{} >{};
    constexpr size_t test_size = 1u << 12;
    auto             original_intervals =
        decltype(detail::computeLocalDofIntervals(std::declval< MeshPartition >(), dummy_problem_def)){};
    original_intervals.reserve(test_size);
    std::generate_n(begin(original_intervals), test_size, [prng = std::mt19937{std::random_device{}()}]() mutable {
        std::uniform_int_distribution< n_id_t >             delim_dist{};
        std::uniform_int_distribution< unsigned long long > field_dist{};
        return std::make_pair(std::array< n_id_t, 2 >{delim_dist(prng), delim_dist(prng)},
                              std::bitset< n_fields >{field_dist(prng)});
    });

    auto deserialized_intervals = original_intervals;
    deserialized_intervals.clear();
    std::vector< unsigned long long > serial;
    serial.reserve((bitsetNUllongs< n_fields >() + 2) * original_intervals.size());
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
        std::array< unsigned long long, bitsetNUllongs< n_fields >() > cov_serial{};
        std::generate_n(begin(cov_serial), cov_serial.size(), [&] {
            return std::uniform_int_distribution< unsigned long long >{}(prng);
        });
        const auto cov = trimBitset< n_fields >(deserializeBitset(cov_serial));
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
            for (const auto& [lo, hi] = interval.first; auto node : std::views::iota(lo, hi + 1))
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
                                    const auto& [lo, hi] = delim;
                                    return lo > std::exchange(prev, hi);
                                });
        CHECK(intervals_are_disjoint);
        const bool intervals_are_correct = std::ranges::all_of(intervals, [&](const auto& interval) {
            const auto& [delims, cov] = interval;
            const auto& [lo, hi]      = delims;
            return std::ranges::all_of(std::views::iota(lo, hi + 1), [&](auto node) { return node_covs[node] == cov; });
        });
        CHECK(intervals_are_correct);
    }
}

TEST_CASE("Node to DOF", "[dof]")
{
    constexpr std::array node_dist = {0., 1., 2., 3., 4.};
    constexpr size_t     n_parts   = 4;
    auto                 mesh      = makeCubeMesh(node_dist);
    auto&                p0        = mesh.getPartitions()[0];
    constexpr size_t     n_fields  = 3;

    SECTION("Non-contiguous DOF distribution")
    {
        detail::node_interval_vector_t< n_fields > dof_intervals;
        n_id_t i1_begin = 0, i1_end = node_dist.size() * node_dist.size() - 1, i2_begin = i1_end + 1,
               i2_end = p0.getNodes().size() - 1;
        std::bitset< n_fields > i1_cov{0b110ull}, i2_cov{0b101ull};
        dof_intervals.emplace_back(std::array{i1_begin, i1_end}, i1_cov);
        dof_intervals.emplace_back(std::array{i2_begin, i2_end}, i2_cov);

        const auto check_dofs = [&](const std::vector< n_id_t >& nodes, const NodeToGlobalDofMap< n_fields >& map) {
            for (auto node : nodes)
            {
                const auto  computed_dofs = map(node);
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
            const auto map = NodeToGlobalDofMap{p0, dof_intervals};
            CHECK_FALSE(map.isContiguous());
            check_dofs(p0.getNodes(), map);
        }

        SECTION("Partitioned")
        {
            mesh                = partitionMesh(mesh, n_parts, {});
            bool not_contiguous = false;
            for (const auto& part : mesh.getPartitions())
            {
                const auto map = NodeToGlobalDofMap{part, dof_intervals};
                not_contiguous |= not map.isContiguous();
                check_dofs(part.getNodes(), map);
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

        const auto check_dofs = [&](const std::vector< n_id_t >& nodes, const NodeToGlobalDofMap< n_fields >& map) {
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
            const auto map = NodeToGlobalDofMap{p0, dof_intervals};
            CHECK(map.isContiguous());
            check_dofs(p0.getNodes(), map);
        }

        SECTION("Partitioned")
        {
            mesh = partitionMesh(mesh, n_parts, {});
            for (const auto& part : mesh.getPartitions())
            {
                const auto map = NodeToGlobalDofMap{part, dof_intervals};
                CHECK(map.isContiguous());
                check_dofs(part.getNodes(), map);
                check_dofs(part.getGhostNodes(), map);
            }
        }
    }
}
