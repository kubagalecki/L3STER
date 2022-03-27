#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/BitsetManip.hpp"
#include "l3ster/util/Common.hpp"
#include "l3ster/util/ConstexprVector.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/Meta.hpp"
#include "l3ster/util/MetisUtils.hpp"
#include "l3ster/util/SetStackSize.hpp"

#include "MakeRandomVector.hpp"

#include "catch2/catch.hpp"
#include "tbb/tbb.h"

#include <random>
#include <ranges>

using namespace lstr;

static consteval auto getConstexprVecParams(int size, int cap)
{
    ConstexprVector< int > v;
    for (int i = 0; i < cap; ++i)
        v.pushBack(i);
    for (int i = cap; i > size; --i)
        v.popBack();
    return std::make_pair(v.size(), v.capacity());
}

static consteval bool checkConstexprVecStorage()
{
    ConstexprVector< ConstexprVector< int > > v(3, ConstexprVector< int >{0, 1, 2});
    bool                                      ret = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ret &= v[i][j] == j;
    return ret;
}

static consteval auto checkConstexprVectorReserve(int cap)
{
    ConstexprVector< int > v;
    v.reserve(cap);
    return std::make_pair(v.size(), v.capacity());
}

static consteval bool checkConstexprVectorIters()
{
    ConstexprVector< int > v;
    constexpr int          size = 10;
    v.reserve(size);
    for (int i = 0; i < size; ++i)
        v.pushBack(i + 1);

    const auto forward_result  = std::accumulate(v.begin(), v.end(), 0);
    const auto backward_result = std::accumulate(v.rbegin(), v.rend(), 0);
    return forward_result == backward_result && forward_result == (size + 1) * size / 2;
}

TEST_CASE("Constexpr vector", "[util]")
{
    SECTION("Size & capcity")
    {
        constexpr int  size = 3, cap = 8;
        constexpr auto vec_params             = getConstexprVecParams(size, cap);
        const auto& [size_result, cap_result] = vec_params;
        CHECK(size_result == size);
        CHECK(cap_result >= cap);
    }

    SECTION("Value storage")
    {
        constexpr bool result = checkConstexprVecStorage();
        CHECK(result);
    }

    SECTION("Reserve capacity")
    {
        constexpr int  cap                    = 10;
        constexpr auto vec_params             = checkConstexprVectorReserve(cap);
        const auto& [size_result, cap_result] = vec_params;
        CHECK(size_result == 0);
        CHECK(cap_result == cap);
    }

    SECTION("Iterators consistent forward and backward")
    {
        constexpr bool result = checkConstexprVectorIters();
        CHECK(result);
    }
}

TEST_CASE("Stack size manipulation", "[util]")
{
    SECTION("Increase stack size by 1")
    {
        const auto& [initial, max] = getStackSize();
        CHECK(initial <= max);
        setMinStackSize(initial + 1);
        const auto& [current, ignore] = getStackSize();
        CHECK(initial + 1 == current);
    }

    SECTION("Increse beyond limit")
    {
        const auto& [initial, max] = getStackSize();
        if (max < std::numeric_limits< std::decay_t< decltype(max) > >::max())
        {
            CHECK_THROWS(setMinStackSize(max + 1ul));
            const auto& [current, ignore] = getStackSize();
            CHECK(initial == current);
        }
    }

    SECTION("Decreasing is a noop")
    {
        const auto& [initial, max] = getStackSize();
        setMinStackSize(initial - 1);
        const auto& [current, ignore] = getStackSize();
        CHECK(initial == current);
    }
}

TEMPLATE_TEST_CASE("Bitset (de-)serialization",
                   "[util]",
                   ConstexprValue< 10u >,
                   ConstexprValue< 64u >,
                   ConstexprValue< 100u >,
                   ConstexprValue< 128u >,
                   ConstexprValue< 257u >)
{
    constexpr auto size               = TestType::value;
    constexpr auto n_runs             = 1 << 8;
    constexpr auto make_random_bitset = [] {
        std::bitset< size >                    retval;
        std::uniform_int_distribution< short > dist{0, 1};
        std::mt19937                           prng{std::random_device{}()};
        for (unsigned i = 0; i < size; ++i)
            retval[i] = dist(prng);
        return retval;
    };
    for (int i = 0; i < n_runs; ++i)
    {
        const auto test_data = make_random_bitset();
        const auto result    = trimBitset< size >(deserializeBitset(serializeBitset(test_data)));
        CHECK(test_data == result);
    }
}

TEST_CASE("MetisGraphWrapper", "[util]")
{
    // Test data is a full graph of size n_nodes
    constexpr ptrdiff_t n_nodes     = 10;
    constexpr ptrdiff_t n_node_nbrs = n_nodes - 1;

    auto node_adjcncy_inds = static_cast< idx_t* >(malloc(sizeof(idx_t) * (n_nodes + 1)));
    auto node_adjcncy      = static_cast< idx_t* >(malloc(sizeof(idx_t) * n_nodes * n_node_nbrs));

    for (ptrdiff_t i = 0; i <= n_nodes; ++i)
        node_adjcncy_inds[i] = i * n_node_nbrs;
    for (ptrdiff_t i = 0; i < n_nodes; ++i)
    {
        auto base = node_adjcncy_inds[i];
        for (ptrdiff_t j = 0; j < i; ++j)
            node_adjcncy[base + j] = j;
        for (ptrdiff_t j = i + 1; j < n_nodes; ++j)
            node_adjcncy[base + j - 1] = j;
    }

    MetisGraphWrapper test_obj1{node_adjcncy_inds, node_adjcncy, n_nodes};
    auto              test_obj2{std::move(test_obj1)};
    auto              test_obj3 = test_obj2;

    test_obj3 = test_obj2;
    test_obj3 = test_obj3;

    auto test_obj4 = test_obj3;
    test_obj3      = std::move(test_obj4);
    test_obj3      = std::move(test_obj3);

    CHECK(std::ranges::equal(test_obj2.getAdjncy(), test_obj3.getAdjncy()));
    CHECK(std::ranges::equal(test_obj2.getXadj(), test_obj3.getXadj()));
}

TEST_CASE("Consecutive reduce algo", "[util]")
{
    SECTION("Default args")
    {
        std::vector v{1, 1, 1, 2, 3, 3, 4, 5, 5, 5};
        v.erase(reduceConsecutive(v).begin(), v.end());
        REQUIRE(v.size() == 5);
        CHECK(v[0] == 3);
        CHECK(v[1] == 2);
        CHECK(v[2] == 6);
        CHECK(v[3] == 4);
        CHECK(v[4] == 15);
    }

    SECTION("Custom comp & reduce")
    {
        std::vector v{std::pair{1, 2},
                      std::pair{4, -1},
                      std::pair{11, -8},
                      std::pair{2, 1},
                      std::pair{2, 0},
                      std::pair{0, 1},
                      std::pair{42, 13},
                      std::pair{54, 1}};
        const auto  cmp = [](const auto& p1, const auto& p2) {
            return p1.first + p1.second == p2.first + p2.second;
        };
        v.erase(reduceConsecutive(v,
                                  cmp,
                                  [](const auto& p1, const auto& p2) {
                                      return std::make_pair(std::abs(p1.first) + std::abs(p2.first),
                                                            std::abs(p1.second) + std::abs(p2.second));
                                  })
                    .begin(),
                v.end());
        REQUIRE(v.size() == 4);
        CHECK(v[0].first == 18);
        CHECK(v[0].second == 12);
        CHECK(v[1].first == 2);
        CHECK(v[1].second == 0);
        CHECK(v[2].first == 0);
        CHECK(v[2].second == 1);
        CHECK(v[3].first == 96);
        CHECK(v[3].second == 14);
    }

    SECTION("Algebraic sequence")
    {
        std::vector v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42};
        v.erase(reduceConsecutive(
                    v, [](int e1, int e2) { return e1 + 1 == e2; }, [](int e1, int e2) { return std::max(e1, e2); })
                    .begin(),
                v.end());
        REQUIRE(v.size() == 2);
        CHECK(v[0] == 10);
        CHECK(v[1] == 42);
    }

    SECTION("Interval reduction")
    {
        std::vector v{std::pair{1, 2},
                      std::pair{2, 3},
                      std::pair{3, 4},
                      std::pair{4, 5},
                      std::pair{5, 6},
                      std::pair{6, 7},
                      std::pair{7, 8},
                      std::pair{8, 9}};
        v.erase(reduceConsecutive(
                    v,
                    [](const auto& p1, const auto& p2) { return p1.second == p2.first; },
                    [](const auto& p1, const auto& p2) { return std::make_pair(p1.first, p2.second); })
                    .begin(),
                v.end());
        REQUIRE(v.size() == 1);
        CHECK(v[0].first == 1);
        CHECK(v[0].second == 9);
    }

    SECTION("Empty")
    {
        std::vector< int > v;
        v.erase(reduceConsecutive(v).begin(), v.end());
        REQUIRE(v.size() == 0);
    }
}

TEST_CASE("Dynamic bitset", "[util]")
{
    constexpr auto make_set_inds = [](size_t size) {
        auto retval = makeRandomVector< size_t >(size / 2, 0, size - 1);
        std::ranges::sort(retval);
        auto del_range = std::ranges::unique(retval);
        retval.erase(del_range.begin(), del_range.end());
        return retval;
    };
    constexpr auto make_reset_inds = [](size_t size, const std::vector< size_t >& set_inds) {
        std::vector< size_t > retval(size - set_inds.size());
        std::ranges::set_difference(std::views::iota(size_t{0}, size), set_inds, begin(retval));
        return retval;
    };

    constexpr auto check_are_set = [](const DynamicBitset& bitset, std::ranges::range auto&& inds) {
        constexpr auto test_as_const = [](const DynamicBitset& bs, size_t pos) {
            return bs[pos];
        };
        for (auto i : inds)
        {
            CHECK(bitset[i]);
            CHECK(test_as_const(bitset, i));
        }
    };
    constexpr auto check_are_reset = [](const DynamicBitset& bitset, std::ranges::range auto&& inds) {
        constexpr auto test_as_const = [](const DynamicBitset& bs, size_t pos) {
            return bs[pos];
        };
        for (auto i : inds)
        {
            CHECK_FALSE(bitset[i]);
            CHECK_FALSE(test_as_const(bitset, i));
        }
    };

    constexpr std::array sizes{10, 64, 128, 10001};

    SECTION("Single-threaded")
    {
        DynamicBitset bitset;
        for (size_t size : sizes)
        {
            bitset.resize(size);
            for (auto i : std::views::iota(size_t{0}, bitset.size()))
                bitset.reset(i);

            const auto set_inds   = make_set_inds(size);
            const auto reset_inds = make_reset_inds(size, set_inds);

            const auto check_set_correctly = [&] {
                check_are_set(bitset, set_inds);
                check_are_reset(bitset, reset_inds);
                CHECK(bitset.count() == set_inds.size());
            };

            for (auto i : set_inds)
                bitset.set(i);
            check_set_correctly();

            for (auto i : std::views::iota(size_t{0}, bitset.size()))
                bitset.flip(i);
            check_are_set(bitset, reset_inds);
            check_are_reset(bitset, set_inds);

            for (auto i : reset_inds)
                bitset.reset(i);
            check_are_reset(bitset, std::views::iota(size_t{0}, bitset.size()));
            CHECK(bitset.count() == 0);

            for (auto i : set_inds)
                bitset[i] = true;
            check_set_correctly();
        }
    }

    SECTION("Subview")
    {
        for (size_t size : sizes)
        {
            {
                const DynamicBitset bitset{size};
                const auto          subview = bitset.getSubView(1, size / 2);
                CHECK(subview.count() == 0);
                CHECK(bitset.getSubView(0, 0).count() == 0);
                CHECK_FALSE(subview[0]);
            }
            {
                DynamicBitset bitset{size};
                auto          subview = bitset.getSubView(1, size / 2);
                for (size_t i = 0; i < subview.size(); ++i)
                {
                    subview.set(i);
                    CHECK(subview.test(i));
                    CHECK(subview[i]);
                    subview.reset(i);
                    CHECK_FALSE(subview.test(i));
                    subview.flip(i);
                    CHECK(subview.test(i));
                    subview.assign(i, false);
                    CHECK_FALSE(subview.test(i));
                    subview[i] = true;
                    CHECK(subview.test(i));
                }
                CHECK(subview.count() == bitset.count());
            }
        }
    }

    SECTION("Multi-threaded")
    {
        for (size_t size : sizes)
        {
            DynamicBitset bitset{size};
            auto          atomic_view = bitset.getAtomicView();
            const auto    set_inds    = make_set_inds(size);
            const auto    reset_inds  = make_reset_inds(size, set_inds);

            tbb::parallel_for(tbb::blocked_range< size_t >{0, set_inds.size()},
                              [&](const tbb::blocked_range< size_t >& range) {
                                  for (size_t i = range.begin(); i != range.end(); ++i)
                                      atomic_view.set(set_inds[i], std::memory_order_relaxed);
                              });
            check_are_set(bitset, set_inds);
            check_are_reset(bitset, reset_inds);
            std::atomic_flag ok{true};
            tbb::parallel_for(tbb::blocked_range< size_t >{0, set_inds.size()},
                              [&](const tbb::blocked_range< size_t >& range) {
                                  for (size_t i = range.begin(); i != range.end(); ++i)
                                      if (not atomic_view.test(set_inds[i]))
                                          ok.clear(std::memory_order_relaxed);
                              });
            CHECK(ok.test());

            tbb::parallel_for(tbb::blocked_range< size_t >{0, bitset.size()},
                              [&](const tbb::blocked_range< size_t >& range) {
                                  for (size_t i = range.begin(); i != range.end(); ++i)
                                      atomic_view.flip(i, std::memory_order_relaxed);
                              });
            check_are_set(bitset, reset_inds);
            check_are_reset(bitset, set_inds);

            tbb::parallel_for(tbb::blocked_range< size_t >{0, reset_inds.size()},
                              [&](const tbb::blocked_range< size_t >& range) {
                                  for (size_t i = range.begin(); i != range.end(); ++i)
                                      atomic_view.reset(reset_inds[i], std::memory_order_relaxed);
                              });
            CHECK(bitset.count() == 0);
        }
    }
}
