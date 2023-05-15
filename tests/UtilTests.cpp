#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Base64.hpp"
#include "l3ster/util/BitsetManip.hpp"
#include "l3ster/util/Common.hpp"
#include "l3ster/util/ConstexprRefStableCollection.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/HwlocWrapper.hpp"
#include "l3ster/util/IndexMap.hpp"
#include "l3ster/util/Meta.hpp"
#include "l3ster/util/MetisUtils.hpp"
#include "l3ster/util/ScopeGuards.hpp"
#include "l3ster/util/SetStackSize.hpp"
#include "l3ster/util/StaticVector.hpp"
#include "l3ster/util/TbbUtils.hpp"

#include "MakeRandomVector.hpp"
#include "TestDataPath.h"

#include "catch2/catch.hpp"
#include "tbb/tbb.h"

#include <algorithm>
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
        constexpr int size = 3, cap = 8;
        const auto [size_result, cap_result] = getConstexprVecParams(size, cap);
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
        constexpr int cap                    = 10;
        const auto [size_result, cap_result] = checkConstexprVectorReserve(cap);
        CHECK(size_result == 0);
        CHECK(cap_result == cap);
    }

    SECTION("Iterators consistent forward and backward")
    {
        constexpr bool result = checkConstexprVectorIters();
        CHECK(result);
    }
}

static consteval auto checkRSC()
{
    ConstexprRefStableCollection< size_t > nums;
    for (size_t i = 1; i < nums.block_size * 2; ++i)
        nums.push(i);
    std::ranges::sort(nums | std::views::reverse);
    return std::ranges::is_sorted(nums, std::greater<>{});
}

TEST_CASE("Constexpr ref-stable collection", "[util]")
{
    static_assert(checkRSC());
}

TEST_CASE("Stack size manipulation", "[util]")
{
    SECTION("Increase stack size by 1")
    {
        const auto [initial, max] = util::detail::getStackSize();
        CHECK(initial <= max);
        util::detail::setMinStackSize(initial + 1);
        const auto [current, ignore] = util::detail::getStackSize();
        CHECK(initial + 1 == current);
    }

    SECTION("Increse beyond limit")
    {
        const auto [initial, max] = util::detail::getStackSize();
        if (max < std::numeric_limits< std::decay_t< decltype(max) > >::max())
        {
            CHECK_THROWS(util::detail::setMinStackSize(max + 1ul));
            const auto [current, ignore] = util::detail::getStackSize();
            CHECK(initial == current);
        }
    }

    SECTION("Decreasing is a noop")
    {
        const auto [initial, max] = util::detail::getStackSize();
        util::detail::setMinStackSize(initial - 1);
        const auto [current, ignore] = util::detail::getStackSize();
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
    constexpr static auto size               = TestType::value;
    constexpr auto        n_runs             = 1 << 8;
    constexpr auto        make_random_bitset = [] {
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
    constexpr idx_t n_nodes     = 10;
    constexpr idx_t n_node_nbrs = n_nodes - 1;

    auto node_adjcncy_inds = static_cast< idx_t* >(malloc(sizeof(idx_t) * (n_nodes + 1)));
    auto node_adjcncy      = static_cast< idx_t* >(malloc(sizeof(idx_t) * n_nodes * n_node_nbrs));

    for (idx_t i = 0; i <= n_nodes; ++i)
        node_adjcncy_inds[i] = i * n_node_nbrs;
    for (idx_t i = 0; i < n_nodes; ++i)
    {
        auto base = node_adjcncy_inds[i];
        for (idx_t j = 0; j < i; ++j)
            node_adjcncy[base + j] = j;
        for (idx_t j = i + 1; j < n_nodes; ++j)
            node_adjcncy[base + j - 1] = j;
    }

    auto test_obj1 = util::metis::GraphWrapper{node_adjcncy_inds, node_adjcncy, n_nodes};
    auto test_obj2{std::move(test_obj1)};
    auto test_obj3 = test_obj2;

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

TEST_CASE("Index map", "[util]")
{
    constexpr size_t           size = 42, base = 10;
    std::array< size_t, size > vals;
    std::ranges::generate(vals, [base = base]() mutable { return base++; });
    const auto map = IndexMap{vals};
    for (size_t i : std::views::iota(base, base + size))
        CHECK(map(i) == i - 10);
}

TEST_CASE("Base64 encoding", "[util]")
{
    constexpr auto text =
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore "
        "magna aliqua. Etiam dignissim diam quis enim lobortis scelerisque fermentum. Mi quis hendrerit dolor magna. "
        "Pellentesque diam volutpat commodo sed. Diam sollicitudin tempor id eu nisl nunc mi ipsum. Eget mauris "
        "pharetra et ultrices neque ornare aenean. Amet luctus venenatis lectus magna fringilla urna porttitor rhoncus "
        "dolor. Donec pretium vulputate sapien nec sagittis aliquam. Non odio euismod lacinia at quis risus sed "
        "vulputate. Dui accumsan sit amet nulla facilisi. Vitae ultricies leo integer malesuada nunc vel risus "
        "commodo. Ut venenatis tellus in metus vulputate eu scelerisque. Imperdiet dui accumsan sit amet nulla "
        "facilisi. Velit sed ullamcorper morbi tincidunt. Nisi vitae suscipit tellus mauris a diam maecenas. "
        "Adipiscing elit duis tristique sollicitudin nibh sit amet commodo. Nulla at volutpat diam ut venenatis tellus "
        "in metus vulputate.";
    constexpr auto encoded_expected =
        "TG9yZW0gaXBzdW0gZG9sb3Igc2l0IGFtZXQsIGNvbnNlY3RldHVyIGFkaXBpc2NpbmcgZWxpdCwgc2VkIGRvIGVpdXNtb2QgdGVtcG9yIGluY2"
        "lkaWR1bnQgdXQgbGFib3JlIGV0IGRvbG9yZSBtYWduYSBhbGlxdWEuIEV0aWFtIGRpZ25pc3NpbSBkaWFtIHF1aXMgZW5pbSBsb2JvcnRpcyBz"
        "Y2VsZXJpc3F1ZSBmZXJtZW50dW0uIE1pIHF1aXMgaGVuZHJlcml0IGRvbG9yIG1hZ25hLiBQZWxsZW50ZXNxdWUgZGlhbSB2b2x1dHBhdCBjb2"
        "1tb2RvIHNlZC4gRGlhbSBzb2xsaWNpdHVkaW4gdGVtcG9yIGlkIGV1IG5pc2wgbnVuYyBtaSBpcHN1bS4gRWdldCBtYXVyaXMgcGhhcmV0cmEg"
        "ZXQgdWx0cmljZXMgbmVxdWUgb3JuYXJlIGFlbmVhbi4gQW1ldCBsdWN0dXMgdmVuZW5hdGlzIGxlY3R1cyBtYWduYSBmcmluZ2lsbGEgdXJuYS"
        "Bwb3J0dGl0b3IgcmhvbmN1cyBkb2xvci4gRG9uZWMgcHJldGl1bSB2dWxwdXRhdGUgc2FwaWVuIG5lYyBzYWdpdHRpcyBhbGlxdWFtLiBOb24g"
        "b2RpbyBldWlzbW9kIGxhY2luaWEgYXQgcXVpcyByaXN1cyBzZWQgdnVscHV0YXRlLiBEdWkgYWNjdW1zYW4gc2l0IGFtZXQgbnVsbGEgZmFjaW"
        "xpc2kuIFZpdGFlIHVsdHJpY2llcyBsZW8gaW50ZWdlciBtYWxlc3VhZGEgbnVuYyB2ZWwgcmlzdXMgY29tbW9kby4gVXQgdmVuZW5hdGlzIHRl"
        "bGx1cyBpbiBtZXR1cyB2dWxwdXRhdGUgZXUgc2NlbGVyaXNxdWUuIEltcGVyZGlldCBkdWkgYWNjdW1zYW4gc2l0IGFtZXQgbnVsbGEgZmFjaW"
        "xpc2kuIFZlbGl0IHNlZCB1bGxhbWNvcnBlciBtb3JiaSB0aW5jaWR1bnQuIE5pc2kgdml0YWUgc3VzY2lwaXQgdGVsbHVzIG1hdXJpcyBhIGRp"
        "YW0gbWFlY2VuYXMuIEFkaXBpc2NpbmcgZWxpdCBkdWlzIHRyaXN0aXF1ZSBzb2xsaWNpdHVkaW4gbmliaCBzaXQgYW1ldCBjb21tb2RvLiBOdW"
        "xsYSBhdCB2b2x1dHBhdCBkaWFtIHV0IHZlbmVuYXRpcyB0ZWxsdXMgaW4gbWV0dXMgdnVscHV0YXRlLg==";
    constexpr auto encoded_expected_drop1 =
        "b3JlbSBpcHN1bSBkb2xvciBzaXQgYW1ldCwgY29uc2VjdGV0dXIgYWRpcGlzY2luZyBlbGl0LCBzZWQgZG8gZWl1c21vZCB0ZW1wb3IgaW5jaW"
        "RpZHVudCB1dCBsYWJvcmUgZXQgZG9sb3JlIG1hZ25hIGFsaXF1YS4gRXRpYW0gZGlnbmlzc2ltIGRpYW0gcXVpcyBlbmltIGxvYm9ydGlzIHNj"
        "ZWxlcmlzcXVlIGZlcm1lbnR1bS4gTWkgcXVpcyBoZW5kcmVyaXQgZG9sb3IgbWFnbmEuIFBlbGxlbnRlc3F1ZSBkaWFtIHZvbHV0cGF0IGNvbW"
        "1vZG8gc2VkLiBEaWFtIHNvbGxpY2l0dWRpbiB0ZW1wb3IgaWQgZXUgbmlzbCBudW5jIG1pIGlwc3VtLiBFZ2V0IG1hdXJpcyBwaGFyZXRyYSBl"
        "dCB1bHRyaWNlcyBuZXF1ZSBvcm5hcmUgYWVuZWFuLiBBbWV0IGx1Y3R1cyB2ZW5lbmF0aXMgbGVjdHVzIG1hZ25hIGZyaW5naWxsYSB1cm5hIH"
        "BvcnR0aXRvciByaG9uY3VzIGRvbG9yLiBEb25lYyBwcmV0aXVtIHZ1bHB1dGF0ZSBzYXBpZW4gbmVjIHNhZ2l0dGlzIGFsaXF1YW0uIE5vbiBv"
        "ZGlvIGV1aXNtb2QgbGFjaW5pYSBhdCBxdWlzIHJpc3VzIHNlZCB2dWxwdXRhdGUuIER1aSBhY2N1bXNhbiBzaXQgYW1ldCBudWxsYSBmYWNpbG"
        "lzaS4gVml0YWUgdWx0cmljaWVzIGxlbyBpbnRlZ2VyIG1hbGVzdWFkYSBudW5jIHZlbCByaXN1cyBjb21tb2RvLiBVdCB2ZW5lbmF0aXMgdGVs"
        "bHVzIGluIG1ldHVzIHZ1bHB1dGF0ZSBldSBzY2VsZXJpc3F1ZS4gSW1wZXJkaWV0IGR1aSBhY2N1bXNhbiBzaXQgYW1ldCBudWxsYSBmYWNpbG"
        "lzaS4gVmVsaXQgc2VkIHVsbGFtY29ycGVyIG1vcmJpIHRpbmNpZHVudC4gTmlzaSB2aXRhZSBzdXNjaXBpdCB0ZWxsdXMgbWF1cmlzIGEgZGlh"
        "bSBtYWVjZW5hcy4gQWRpcGlzY2luZyBlbGl0IGR1aXMgdHJpc3RpcXVlIHNvbGxpY2l0dWRpbiBuaWJoIHNpdCBhbWV0IGNvbW1vZG8uIE51bG"
        "xhIGF0IHZvbHV0cGF0IGRpYW0gdXQgdmVuZW5hdGlzIHRlbGx1cyBpbiBtZXR1cyB2dWxwdXRhdGUu";
    constexpr auto encoded_expected_drop2 =
        "cmVtIGlwc3VtIGRvbG9yIHNpdCBhbWV0LCBjb25zZWN0ZXR1ciBhZGlwaXNjaW5nIGVsaXQsIHNlZCBkbyBlaXVzbW9kIHRlbXBvciBpbmNpZG"
        "lkdW50IHV0IGxhYm9yZSBldCBkb2xvcmUgbWFnbmEgYWxpcXVhLiBFdGlhbSBkaWduaXNzaW0gZGlhbSBxdWlzIGVuaW0gbG9ib3J0aXMgc2Nl"
        "bGVyaXNxdWUgZmVybWVudHVtLiBNaSBxdWlzIGhlbmRyZXJpdCBkb2xvciBtYWduYS4gUGVsbGVudGVzcXVlIGRpYW0gdm9sdXRwYXQgY29tbW"
        "9kbyBzZWQuIERpYW0gc29sbGljaXR1ZGluIHRlbXBvciBpZCBldSBuaXNsIG51bmMgbWkgaXBzdW0uIEVnZXQgbWF1cmlzIHBoYXJldHJhIGV0"
        "IHVsdHJpY2VzIG5lcXVlIG9ybmFyZSBhZW5lYW4uIEFtZXQgbHVjdHVzIHZlbmVuYXRpcyBsZWN0dXMgbWFnbmEgZnJpbmdpbGxhIHVybmEgcG"
        "9ydHRpdG9yIHJob25jdXMgZG9sb3IuIERvbmVjIHByZXRpdW0gdnVscHV0YXRlIHNhcGllbiBuZWMgc2FnaXR0aXMgYWxpcXVhbS4gTm9uIG9k"
        "aW8gZXVpc21vZCBsYWNpbmlhIGF0IHF1aXMgcmlzdXMgc2VkIHZ1bHB1dGF0ZS4gRHVpIGFjY3Vtc2FuIHNpdCBhbWV0IG51bGxhIGZhY2lsaX"
        "NpLiBWaXRhZSB1bHRyaWNpZXMgbGVvIGludGVnZXIgbWFsZXN1YWRhIG51bmMgdmVsIHJpc3VzIGNvbW1vZG8uIFV0IHZlbmVuYXRpcyB0ZWxs"
        "dXMgaW4gbWV0dXMgdnVscHV0YXRlIGV1IHNjZWxlcmlzcXVlLiBJbXBlcmRpZXQgZHVpIGFjY3Vtc2FuIHNpdCBhbWV0IG51bGxhIGZhY2lsaX"
        "NpLiBWZWxpdCBzZWQgdWxsYW1jb3JwZXIgbW9yYmkgdGluY2lkdW50LiBOaXNpIHZpdGFlIHN1c2NpcGl0IHRlbGx1cyBtYXVyaXMgYSBkaWFt"
        "IG1hZWNlbmFzLiBBZGlwaXNjaW5nIGVsaXQgZHVpcyB0cmlzdGlxdWUgc29sbGljaXR1ZGluIG5pYmggc2l0IGFtZXQgY29tbW9kby4gTnVsbG"
        "EgYXQgdm9sdXRwYXQgZGlhbSB1dCB2ZW5lbmF0aXMgdGVsbHVzIGluIG1ldHVzIHZ1bHB1dGF0ZS4=";
    constexpr auto encoded_expected_takebut1 =
        "TG9yZW0gaXBzdW0gZG9sb3Igc2l0IGFtZXQsIGNvbnNlY3RldHVyIGFkaXBpc2NpbmcgZWxpdCwgc2VkIGRvIGVpdXNtb2QgdGVtcG9yIGluY2"
        "lkaWR1bnQgdXQgbGFib3JlIGV0IGRvbG9yZSBtYWduYSBhbGlxdWEuIEV0aWFtIGRpZ25pc3NpbSBkaWFtIHF1aXMgZW5pbSBsb2JvcnRpcyBz"
        "Y2VsZXJpc3F1ZSBmZXJtZW50dW0uIE1pIHF1aXMgaGVuZHJlcml0IGRvbG9yIG1hZ25hLiBQZWxsZW50ZXNxdWUgZGlhbSB2b2x1dHBhdCBjb2"
        "1tb2RvIHNlZC4gRGlhbSBzb2xsaWNpdHVkaW4gdGVtcG9yIGlkIGV1IG5pc2wgbnVuYyBtaSBpcHN1bS4gRWdldCBtYXVyaXMgcGhhcmV0cmEg"
        "ZXQgdWx0cmljZXMgbmVxdWUgb3JuYXJlIGFlbmVhbi4gQW1ldCBsdWN0dXMgdmVuZW5hdGlzIGxlY3R1cyBtYWduYSBmcmluZ2lsbGEgdXJuYS"
        "Bwb3J0dGl0b3IgcmhvbmN1cyBkb2xvci4gRG9uZWMgcHJldGl1bSB2dWxwdXRhdGUgc2FwaWVuIG5lYyBzYWdpdHRpcyBhbGlxdWFtLiBOb24g"
        "b2RpbyBldWlzbW9kIGxhY2luaWEgYXQgcXVpcyByaXN1cyBzZWQgdnVscHV0YXRlLiBEdWkgYWNjdW1zYW4gc2l0IGFtZXQgbnVsbGEgZmFjaW"
        "xpc2kuIFZpdGFlIHVsdHJpY2llcyBsZW8gaW50ZWdlciBtYWxlc3VhZGEgbnVuYyB2ZWwgcmlzdXMgY29tbW9kby4gVXQgdmVuZW5hdGlzIHRl"
        "bGx1cyBpbiBtZXR1cyB2dWxwdXRhdGUgZXUgc2NlbGVyaXNxdWUuIEltcGVyZGlldCBkdWkgYWNjdW1zYW4gc2l0IGFtZXQgbnVsbGEgZmFjaW"
        "xpc2kuIFZlbGl0IHNlZCB1bGxhbWNvcnBlciBtb3JiaSB0aW5jaWR1bnQuIE5pc2kgdml0YWUgc3VzY2lwaXQgdGVsbHVzIG1hdXJpcyBhIGRp"
        "YW0gbWFlY2VuYXMuIEFkaXBpc2NpbmcgZWxpdCBkdWlzIHRyaXN0aXF1ZSBzb2xsaWNpdHVkaW4gbmliaCBzaXQgYW1ldCBjb21tb2RvLiBOdW"
        "xsYSBhdCB2b2x1dHBhdCBkaWFtIHV0IHZlbmVuYXRpcyB0ZWxsdXMgaW4gbWV0dXMgdnVscHV0YXRl";
    constexpr auto encoded_expected_takebut2 =
        "TG9yZW0gaXBzdW0gZG9sb3Igc2l0IGFtZXQsIGNvbnNlY3RldHVyIGFkaXBpc2NpbmcgZWxpdCwgc2VkIGRvIGVpdXNtb2QgdGVtcG9yIGluY2"
        "lkaWR1bnQgdXQgbGFib3JlIGV0IGRvbG9yZSBtYWduYSBhbGlxdWEuIEV0aWFtIGRpZ25pc3NpbSBkaWFtIHF1aXMgZW5pbSBsb2JvcnRpcyBz"
        "Y2VsZXJpc3F1ZSBmZXJtZW50dW0uIE1pIHF1aXMgaGVuZHJlcml0IGRvbG9yIG1hZ25hLiBQZWxsZW50ZXNxdWUgZGlhbSB2b2x1dHBhdCBjb2"
        "1tb2RvIHNlZC4gRGlhbSBzb2xsaWNpdHVkaW4gdGVtcG9yIGlkIGV1IG5pc2wgbnVuYyBtaSBpcHN1bS4gRWdldCBtYXVyaXMgcGhhcmV0cmEg"
        "ZXQgdWx0cmljZXMgbmVxdWUgb3JuYXJlIGFlbmVhbi4gQW1ldCBsdWN0dXMgdmVuZW5hdGlzIGxlY3R1cyBtYWduYSBmcmluZ2lsbGEgdXJuYS"
        "Bwb3J0dGl0b3IgcmhvbmN1cyBkb2xvci4gRG9uZWMgcHJldGl1bSB2dWxwdXRhdGUgc2FwaWVuIG5lYyBzYWdpdHRpcyBhbGlxdWFtLiBOb24g"
        "b2RpbyBldWlzbW9kIGxhY2luaWEgYXQgcXVpcyByaXN1cyBzZWQgdnVscHV0YXRlLiBEdWkgYWNjdW1zYW4gc2l0IGFtZXQgbnVsbGEgZmFjaW"
        "xpc2kuIFZpdGFlIHVsdHJpY2llcyBsZW8gaW50ZWdlciBtYWxlc3VhZGEgbnVuYyB2ZWwgcmlzdXMgY29tbW9kby4gVXQgdmVuZW5hdGlzIHRl"
        "bGx1cyBpbiBtZXR1cyB2dWxwdXRhdGUgZXUgc2NlbGVyaXNxdWUuIEltcGVyZGlldCBkdWkgYWNjdW1zYW4gc2l0IGFtZXQgbnVsbGEgZmFjaW"
        "xpc2kuIFZlbGl0IHNlZCB1bGxhbWNvcnBlciBtb3JiaSB0aW5jaWR1bnQuIE5pc2kgdml0YWUgc3VzY2lwaXQgdGVsbHVzIG1hdXJpcyBhIGRp"
        "YW0gbWFlY2VuYXMuIEFkaXBpc2NpbmcgZWxpdCBkdWlzIHRyaXN0aXF1ZSBzb2xsaWNpdHVkaW4gbmliaCBzaXQgYW1ldCBjb21tb2RvLiBOdW"
        "xsYSBhdCB2b2x1dHBhdCBkaWFtIHV0IHZlbmVuYXRpcyB0ZWxsdXMgaW4gbWV0dXMgdnVscHV0YXQ=";
    constexpr auto text_sv                      = std::string_view{text};
    constexpr auto encoded_expected_sv          = std::string_view{encoded_expected};
    constexpr auto encoded_expected_drop1_sv    = std::string_view{encoded_expected_drop1};
    constexpr auto encoded_expected_drop2_sv    = std::string_view{encoded_expected_drop2};
    constexpr auto encoded_expected_takebut1_sv = std::string_view{encoded_expected_takebut1};
    constexpr auto encoded_expected_takebut2_sv = std::string_view{encoded_expected_takebut2};

    std::vector< char > alloc(text_sv.size() * 4u / 3u + 3u);
    const auto          test = [&alloc](auto&& txt, std::string_view expected_b64) {
        const auto bytes_written = encodeAsBase64(txt, alloc.begin());
        const auto encoded       = std::string_view{alloc.data(), bytes_written};
        REQUIRE(std::ranges::size(encoded) == expected_b64.size());
        CHECK(std::ranges::equal(encoded, expected_b64));
    };

    test(text_sv, encoded_expected_sv);
    test(text_sv | std::views::drop(1), encoded_expected_drop1_sv);
    test(text_sv | std::views::drop(2), encoded_expected_drop2_sv);
    test(text_sv | std::views::take(text_sv.size() - 1), encoded_expected_takebut1_sv);
    test(text_sv | std::views::take(text_sv.size() - 2), encoded_expected_takebut2_sv);
    test(text_sv | std::views::drop(3) | std::views::take(2), std::string_view{"ZW0="});

    // Test whether sequential and parallel implementations yield identical results
    constexpr size_t long_text_size = 1ul << 25;
    std::string      long_text(long_text_size, '\0');
    std::ranges::generate(
        long_text,
        [prng = std::mt19937{std::random_device{}()}, dist = std::uniform_int_distribution< int >{0, 255}]() mutable {
            return static_cast< char >(dist(prng));
        });
    std::string long_text_b64_par(long_text_size * 4 / 3 + 4, '\0');
    std::string long_text_b64_seq(long_text_size * 4 / 3 + 4, '\0');
    encodeAsBase64(long_text, long_text_b64_par.begin());
    auto       seq_ptr        = long_text_b64_seq.data();
    const auto long_byte_span = std::as_bytes(std::span{long_text});
    const auto lp             = detail::b64::encB64SerialImpl(long_byte_span, seq_ptr);
    detail::b64::encB64Remainder(long_byte_span.subspan(lp), seq_ptr);
    CHECK(long_text_b64_seq == long_text_b64_par);
}

TEST_CASE("StaticVector", "[util]")
{
    auto vec = util::StaticVector< int, 100 >{};
    CHECK(vec.size() == 0);
    std::generate_n(std::back_inserter(vec),
                    100,
                    [prng = std::mt19937{std::random_device{}()},
                     dist = std::uniform_int_distribution< int >{}]() mutable { return dist(prng); });
    REQUIRE(vec.size() == 100);
    vec.resize(50);
    REQUIRE(vec.size() == 50);
    std::ranges::sort(vec);
    CHECK(vec.front() <= vec.back());
    CHECK(std::ranges::is_sorted(vec));
    vec.erase(std::next(vec.begin(), 10), std::prev(vec.end(), 10));
    REQUIRE(vec.size() == 20);
    vec.resize(10);
    CHECK(std::ranges::is_sorted(vec));
    vec.pop_back();
    REQUIRE(vec.size() == 9);
    const auto vec_copy = vec;
    vec.erase(vec.end(), vec.end());
    REQUIRE(vec.size() == 9);
    CHECK(std::ranges::equal(vec, vec_copy));
    vec.resize(10, 42);
    REQUIRE(vec.size() == 10);
    CHECK(std::ranges::equal(vec | std::views::drop(9), std::views::single(42)));
}

TEST_CASE("getTrueInds", "[util]")
{
    constexpr auto in  = std::array{false, false, false, true};
    constexpr auto out = getTrueInds< in >();
    static_assert(std::ranges::equal(out, std::views::single(3)));
}

TEST_CASE("Non-random access parallel for", "[util]")
{
    constexpr auto n_reps = 3;
    auto           prng   = std::mt19937{std::random_device{}()};
    for (auto size : {1u << 10, 1u << 15, 1u << 18, 1u << 22})
    {
        auto map  = robin_hood::unordered_flat_map< size_t, size_t >{};
        auto dist = std::uniform_int_distribution< size_t >{0, size};

        for (size_t i = 0; i != size; ++i)
            map.emplace(dist(prng), dist(prng));

        for (auto r = 0; r != n_reps; ++r)
        {
            auto parallel_accumulator = std::atomic< size_t >{0};
            util::tbb::parallelFor(map, [&](const auto& map_entry) {
                const auto hash   = std::hash< size_t >{};
                const auto result = hash(hash(map_entry.first) ^ hash(map_entry.second));
                parallel_accumulator.fetch_add(result, std::memory_order_relaxed);
            });
            const auto sequential_sum =
                std::transform_reduce(map.cbegin(), map.cend(), size_t{}, std::plus{}, [](const auto& map_entry) {
                    const auto hash = std::hash< size_t >{};
                    return hash(hash(map_entry.first) ^ hash(map_entry.second));
                });
            CHECK(parallel_accumulator.load() == sequential_sum);
        }
    }
}

TEST_CASE("Hwloc topology info", "[util, hwloc]")
{
    const auto topology = util::hwloc::Topology{};
    REQUIRE_FALSE(topology.isEmpty());
    REQUIRE(L3STER_N_NUMA_NODES == topology.getNNodes());
    REQUIRE(L3STER_N_CORES == topology.getNCores());
    REQUIRE(L3STER_N_HWTHREADS == topology.getNHwThreads());
}

#ifdef _OPENMP
TEST_CASE("OpenMP num threads control", "[util]")
{
    const auto max_threads_initial = omp_get_max_threads();
    {
        const auto max_par_guard = detail::MaxParallelismGuard{1};
        CHECK(omp_get_max_threads() == 1);
    }
    CHECK(omp_get_max_threads() == max_threads_initial);
}
#endif
