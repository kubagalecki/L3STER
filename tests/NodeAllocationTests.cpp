#include "alloc/NodeGlobalMemoryResource.hpp"
#include "util/GlobalResource.hpp"

#include "catch2/catch.hpp"

#include <vector>

TEST_CASE("Node allocation tests", "[hwloc]")
{
    auto&                    topo = lstr::GlobalResource< lstr::HwlocWrapper >::getMaybeUninitialized();
    lstr::NodeGlobalResource alloc{&topo, 0};
    {
        std::pmr::vector< double > v(&alloc);
        v.push_back(3.14);
        v.push_back(3.14);
        CHECK(v.size() == 2);
    }

    constexpr size_t too_big_alloc = std::numeric_limits< size_t >::max() / 2 - 1;
    CHECK_THROWS_AS(alloc.allocate(too_big_alloc), std::bad_alloc);

    std::pmr::unsynchronized_pool_resource res;
    CHECK_FALSE(alloc.is_equal(res));
}
