#include "l3ster.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

#include <algorithm>
#include <atomic>
#include <thread>

TEST_CASE("hwloc topology test", "[hwloc]")
{
    lstr::HwlocWrapper topology{};
    REQUIRE(L3STER_N_NUMA_NODES == topology.getMachineSize());

    lstr::detail::HwlocBitmapRaiiWrapper aggregate_bmp{};
    lstr::detail::HwlocBitmapRaiiWrapper helper_bmp{};
    size_t                               n_cpus = 0;

    // Assert all cpu masks are disjoint and sum up to full machine
    for (size_t node = 0; node < topology.getMachineSize(); ++node)
    {
        for (const auto& cpu_mask : topology.getNodeMasks(node))
        {
            hwloc_bitmap_and(helper_bmp, aggregate_bmp, cpu_mask);
            CHECK(hwloc_bitmap_iszero(helper_bmp));
            hwloc_bitmap_or(aggregate_bmp, aggregate_bmp, cpu_mask);
            ++n_cpus;
        }
    }
    CHECK(hwloc_bitmap_isequal(aggregate_bmp, topology.getMachineMask()));
    CHECK(n_cpus == std::thread::hardware_concurrency());
}

TEST_CASE("hwloc thread-to-core binding test", "[hwloc]")
{
    lstr::HwlocWrapper         toplogy{};
    std::vector< std::thread > thread_pool(std::thread::hardware_concurrency());
    size_t                     node_index = 0, cpu_index = 0;
    std::atomic_bool           hwloc_thread_binding_test_result{true};
    std::ranges::generate(thread_pool, [&] {
        std::thread ret{[&](size_t node, size_t cpu) {
                            // Catch2 doesn't support multithreaded tests, so manual try-catch is needed
                            try
                            {
                                toplogy.bindThreadToCore(node, cpu);
                                const auto obtained_result       = toplogy.getLastThreadLocation();
                                const auto expected_resut        = std::make_pair(node, cpu);
                                hwloc_thread_binding_test_result = obtained_result == expected_resut;
                            }
                            catch (...)
                            {
                                hwloc_thread_binding_test_result = false;
                            }
                        },
                        node_index,
                        cpu_index};
        ++cpu_index;
        if (cpu_index == toplogy.getNodeSize(node_index))
        {
            ++node_index;
            cpu_index = 0;
        }
        return ret;
    });
    std::ranges::for_each(thread_pool, [](std::thread& thread) { thread.join(); });
    CHECK(hwloc_thread_binding_test_result);
}

TEST_CASE("hwloc thread-to-node binding test", "[hwloc]")
{
    lstr::HwlocWrapper         topology;
    std::vector< std::thread > thread_pool(topology.getMachineSize());
    std::atomic_bool           hwloc_thread_binding_test_result{true};
    std::ranges::generate(thread_pool, [&, index = 0u]() mutable {
        return std::thread{[&](size_t node) {
                               try
                               {
                                   topology.bindThreadToNode(node);
                                   const auto exec_location         = topology.getLastThreadLocation();
                                   hwloc_thread_binding_test_result = exec_location.first == node;
                               }
                               catch (...)
                               {
                                   hwloc_thread_binding_test_result = false;
                               }
                           },
                           index++};
    });
    std::ranges::for_each(thread_pool, [](std::thread& thread) { thread.join(); });
    CHECK(hwloc_thread_binding_test_result);
}

TEST_CASE("hwloc memory binding test", "[hwloc]")
{
    lstr::HwlocWrapper topology{};
    constexpr size_t   size = 42 * sizeof(size_t);
    for (size_t node = 0; node < topology.getMachineSize(); ++node)
    {
        volatile auto allocated = static_cast< size_t* >(topology.allocateOnNode(size, node));
        REQUIRE_FALSE(allocated == nullptr);
        allocated[size - 1] = 42u;
        topology.free(allocated, size);
    }
}
