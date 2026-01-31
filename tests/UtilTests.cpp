#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Base64.hpp"
#include "l3ster/util/Common.hpp"
#include "l3ster/util/CrsGraph.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/HwlocWrapper.hpp"
#include "l3ster/util/IO.hpp"
#include "l3ster/util/IndexMap.hpp"
#include "l3ster/util/Meta.hpp"
#include "l3ster/util/MetisUtils.hpp"
#include "l3ster/util/ScopeGuards.hpp"
#include "l3ster/util/Serialization.hpp"
#include "l3ster/util/SetStackSize.hpp"
#include "l3ster/util/SpatialHashTable.hpp"
#include "l3ster/util/StaticVector.hpp"
#include "l3ster/util/TbbUtils.hpp"
#include "l3ster/util/TypeErasedOverload.hpp"
#include "l3ster/util/UniVector.hpp"

#include "MakeRandomVector.hpp"
#include "TestDataPath.h"

#include "catch2/catch.hpp"

#include <algorithm>
#include <random>
#include <ranges>

using namespace lstr;

TEST_CASE("Stack size manipulation", "[util]")
{
    SECTION("Increase stack size by 1")
    {
        const auto [initial, max] = util::detail::getStackSize();
        REQUIRE(initial <= max);
        if (initial < std::numeric_limits< std::decay_t< decltype(initial) > >::max() and initial < max)
        {
            util::detail::setMinStackSize(initial + 1);
            const auto [current, ignore] = util::detail::getStackSize();
            CHECK(initial + 1 == current);
        }
    }

    SECTION("Increse beyond limit")
    {
        const auto [initial, max] = util::detail::getStackSize();
        if (max < std::numeric_limits< std::decay_t< decltype(max) > >::max())
        {
            CHECK_THROWS(util::detail::setMinStackSize(max + 1));
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

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wself-move"
#endif

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
    test_obj3 = test_obj3; // NOLINT

    auto test_obj4 = test_obj3;
    test_obj3      = std::move(test_obj4);
    test_obj3      = std::move(test_obj3); // NOLINT

    CHECK(std::ranges::equal(test_obj2.getAdjncy(), test_obj3.getAdjncy()));
    CHECK(std::ranges::equal(test_obj2.getXadj(), test_obj3.getXadj()));
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

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
        std::ranges::set_difference(std::views::iota(0uz, size), set_inds, begin(retval));
        return retval;
    };

    constexpr auto check_are_set = [](const util::DynamicBitset& bitset, std::ranges::range auto&& inds) {
        constexpr auto test_as_const = [](const util::DynamicBitset& bs, size_t pos) {
            return bs[pos];
        };
        for (auto i : inds)
        {
            CHECK(bitset[i]);
            CHECK(test_as_const(bitset, i));
        }
    };
    constexpr auto check_are_reset = [](const util::DynamicBitset& bitset, std::ranges::range auto&& inds) {
        constexpr auto test_as_const = [](const util::DynamicBitset& bs, size_t pos) {
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
        util::DynamicBitset bitset;
        for (size_t size : sizes)
        {
            bitset.resize(size);
            for (auto i : std::views::iota(0uz, bitset.size()))
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

            for (auto i : std::views::iota(0uz, bitset.size()))
                bitset.flip(i);
            check_are_set(bitset, reset_inds);
            check_are_reset(bitset, set_inds);

            for (auto i : reset_inds)
                bitset.reset(i);
            check_are_reset(bitset, std::views::iota(0uz, bitset.size()));
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
                const util::DynamicBitset bitset{size};
                const auto                subview = bitset.getSubView(1, size / 2);
                CHECK(subview.count() == 0);
                CHECK(bitset.getSubView(0, 0).count() == 0);
                CHECK_FALSE(subview[0]);
            }
            {
                util::DynamicBitset bitset{size};
                auto                subview = bitset.getSubView(1, size / 2);
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
            util::DynamicBitset bitset{size};
            auto                atomic_view = bitset.getAtomicView();
            const auto          set_inds    = make_set_inds(size);
            const auto          reset_inds  = make_reset_inds(size, set_inds);

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
    const auto map = util::IndexMap{vals};
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
        const auto bytes_written = util::encodeAsBase64(txt, alloc.begin());
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
#ifdef DNDEBUG
    constexpr size_t long_text_size = 1ul << 25;
#else
    constexpr size_t long_text_size = 1ul << 20;
#endif
    std::string long_text(long_text_size, '\0');
    std::ranges::generate(
        long_text,
        [prng = std::mt19937{std::random_device{}()}, dist = std::uniform_int_distribution< int >{0, 255}]() mutable {
            return static_cast< char >(dist(prng));
        });
    std::string long_text_b64_par(long_text_size * 4 / 3 + 4, '\0');
    std::string long_text_b64_seq(long_text_size * 4 / 3 + 4, '\0');
    util::encodeAsBase64(long_text, long_text_b64_par.begin());
    auto       seq_ptr        = long_text_b64_seq.data();
    const auto long_byte_span = std::as_bytes(std::span{long_text});
    const auto lp             = util::b64::encB64SerialImpl(long_byte_span, seq_ptr);
    util::b64::encB64Remainder(long_byte_span.subspan(lp), seq_ptr);
    CHECK(long_text_b64_seq == long_text_b64_par);
}

TEST_CASE("Serialization", "[util]")
{
    SECTION("Value")
    {
        constexpr int data = 42;
        std::string   result;
        util::serialize(data, std::back_inserter(result));
        REQUIRE(result.size() == sizeof data);
        const auto deserialized = util::deserialize< int >(result);
        CHECK(data == deserialized);
    }

    SECTION("Text")
    {
        constexpr auto text = std::string_view{"This is some sample text."};
        std::string    result;
        util::serialize(text, std::back_inserter(result));

        constexpr auto szsz = sizeof(size_t);
        REQUIRE(result.size() == text.size() + szsz);
        const auto     result_view = std::string_view{result};
        constexpr auto size        = text.size();
        CHECK(std::ranges::equal(std::span{reinterpret_cast< const char* >(&size), szsz}, result_view.substr(0, szsz)));
        CHECK(result_view.substr(sizeof(size_t)) == text);
    }

    SECTION("Nested ranges")
    {
        const auto  data = std::vector< std::string >{"This ", "is ", "some ", "sample ", "text."};
        std::string result;
        util::serialize(data, std::back_inserter(result));
        const auto deserialized = util::deserialize< std::vector< std::string > >(result);
        CHECK(data == deserialized);
    }

    SECTION("Complex nested types")
    {
        const auto  data = std::map< std::string, std::string >{{"KEY 1", "VAL 1"},
                                                                {"KEY 2", "VAL 2"},
                                                                {"KEY 3", "VAL 3"},
                                                                {"KEY 4", "VAL 4"},
                                                                {"KEY 5", "VAL 5"},
                                                                {"KEY 6", "VAL 6"}};
        std::string result;
        util::serialize(data, std::back_inserter(result));

        const auto deserialized = util::deserialize< std::vector< std::array< std::string, 2 > > >(result);
        CHECK(std::ranges::equal(data, deserialized, [](const auto& pair, const auto& array) {
            return pair.first == array[0] and pair.second == array[1];
        }));
    }

    SECTION("References and views")
    {
        auto        inds = std::vector< std::vector< size_t > >{{1, 0}, {0, 1}};
        auto        strs = std::vector< std::string >{"abc", "def"};
        std::string result;

        auto view = inds | std::views::join | std::views::transform([&](auto& i) { return std::tie(i, strs[i]); });
        util::serialize(view, std::back_inserter(result));

        const auto deserialized = util::deserialize< std::vector< std::pair< size_t, std::string > > >(result);
        REQUIRE(deserialized.size() == static_cast< size_t >(std::ranges::distance(view)));
        CHECK(deserialized[0] == std::make_pair(1, "def"));
        CHECK(deserialized[1] == std::make_pair(0, "abc"));
        CHECK(deserialized[2] == std::make_pair(0, "abc"));
        CHECK(deserialized[3] == std::make_pair(1, "def"));
    }
}

TEST_CASE("StaticVector", "[util]")
{
    SECTION("Ctors")
    {
        SECTION("Default")
        {
            auto vec = util::StaticVector< int, 100 >{};
            REQUIRE(vec.size() == 0);
        }
        SECTION("Array")
        {
            const auto src_arr = std::array{1, 3, 5, 10, 42};
            const auto vec     = util::StaticVector< int, 10 >{src_arr};
            REQUIRE(std::ranges::equal(src_arr, vec));
        }
        SECTION("Range-based")
        {
            const auto src_vec = std::vector{1, 3, 5, 10, 42};

            const auto full_vec = util::StaticVector< int, 10 >{src_vec};
            REQUIRE(std::ranges::equal(src_vec, full_vec));

            auto       even_view = src_vec | std::views::filter([](int i) { return i % 2 == 0; });
            const auto even_vec  = util::StaticVector< int, 10 >{even_view};
            REQUIRE(std::ranges::equal(even_vec, even_view));
        }
    }
    SECTION("Manip")
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
}

TEST_CASE("getTrueInds", "[util]")
{
    constexpr auto in  = std::array{false, false, false, true};
    constexpr auto out = util::getTrueInds< in >();
    static_assert(std::ranges::equal(out, std::views::single(3)));
}

TEST_CASE("Non-random access parallel for", "[util]")
{
#ifdef NDEBUG
    constexpr auto n_reps = 4;
    const auto     sizes  = {1u << 10, 1u << 15, 1u << 18, 1u << 22};
#else
    constexpr auto n_reps = 1;
    const auto     sizes  = {1u << 18, 1u << 12, 1u << 16, 1u << 20};
#endif
    auto prng = std::mt19937{std::random_device{}()};
    for (auto size : sizes)
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
                std::transform_reduce(map.cbegin(), map.cend(), 0uz, std::plus{}, [](const auto& map_entry) {
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
    const auto numa_expected    = L3STER_N_NUMA_NODES;
    const auto cores_expected   = L3STER_N_CORES;
    const auto threads_expected = L3STER_N_HWTHREADS;
    CHECK(numa_expected == topology.getNNodes());
    CHECK(cores_expected == topology.getNCores());
    CHECK(threads_expected == topology.getNHwThreads());
}

TEST_CASE("Num threads control", "[util]")
{
    constexpr auto check_num_threads = [](size_t expected) {
        const auto tbb_threads =
            oneapi::tbb::global_control::active_value(oneapi::tbb::global_control::max_allowed_parallelism);
        CHECK(tbb_threads == expected);
#ifdef _OPENMP
        CHECK(omp_get_max_threads() == static_cast< int >(expected));
#endif
    };

    const auto n_hw_threads = util::GlobalResource< util::hwloc::Topology >::getMaybeUninitialized().getNHwThreads();
    check_num_threads(n_hw_threads);

    {
        const auto guard = util::MaxParallelismGuard{1};
        check_num_threads(1);
    }
    check_num_threads(n_hw_threads);

    {
        const auto guard = util::MaxParallelismGuard{2 * n_hw_threads};
        check_num_threads(n_hw_threads);
    }
    check_num_threads(n_hw_threads);
}

TEST_CASE("Memory-mapping utils", "[utils]")
{
    constexpr size_t size = 1 << 16;
    constexpr size_t ofs = 10, len = 20;
    const auto       filename = std::string("mmap_test.temp");

    auto text = util::ArrayOwner< char >(size);
    std::ranges::generate(text,
                          [prng = std::mt19937{std::random_device{}()},
                           dist = std::uniform_int_distribution< unsigned >(65, 90)]() mutable {
                              return static_cast< char >(dist(prng));
                          });
    text[ofs] = '?';
    auto file = std::ofstream{filename};
    std::ranges::copy(text, std::ostream_iterator< char >(file));
    file.close();

    auto file_mmapped = util::MmappedFile{filename};

    SECTION("mmap")
    {
        REQUIRE(std::ranges::equal(file_mmapped.view(), text));
        CHECK_THROWS(util::MmappedFile{"nonexistent/file/name.txt"});
    }

    SECTION("Stream interface")
    {
        auto bstream = util::MmappedStreambuf(std::move(file_mmapped));
        auto istream = std::istream{&bstream};
        REQUIRE(std::ranges::equal(text, std::views::istream< char >(istream)));
        CHECK_FALSE(istream.good());
    }

    SECTION("Search")
    {
        const auto view      = file_mmapped.view();
        const auto text_view = std::string_view{text.data(), text.size()};
        const auto needle    = view.substr(ofs, len);
        auto       bstream   = util::MmappedStreambuf(std::move(file_mmapped));
        auto       istream   = std::istream{&bstream};
        bstream.skipPast(needle);

        REQUIRE(std::ranges::equal(text_view.substr(ofs + len), std::views::istream< char >(istream)));
        CHECK_FALSE(istream.good());
    }
}

TEST_CASE("Type-erased overload set tests", "[util]")
{
    using namespace std::string_literals;
    using teos_t              = util::TypeErasedOverload< int(int), int(const std::string&) >;
    constexpr auto check_vals = [](const teos_t& teos) {
        CHECK(teos(1) == 0);
        CHECK(teos("str"s) == 1);
    };

    SECTION("Target fits in buffer")
    {
        constexpr int int_val = 0, str_val = 1;
        const auto    process_int = [&](int) {
            return int_val;
        };
        const auto process_string = [&](const std::string&) {
            return str_val;
        };
        const auto overload = util::OverloadSet{process_int, process_string};

        auto te_overload = teos_t{overload};
        check_vals(te_overload);

        auto te2 = std::move(te_overload);
        check_vals(te2);

        int i       = 0;
        te_overload = [&i](const auto&) {
            return i++;
        };
        check_vals(te_overload);
        i = 0;

        teos_t empty;
        CHECK(!empty);
        te2 = std::move(empty);
        CHECK(!te2);
        te2 = std::move(te_overload);
        check_vals(te2);
    }

    SECTION("Target allocated on heap")
    {
        const auto a           = std::array{0, 1, 2, 3, 4};
        const auto process_int = [a = a](int) {
            return a[0];
        };
        const auto process_string = [a = a](const std::string&) {
            return a[1];
        };
        const auto overload = util::OverloadSet{process_int, process_string};

        auto te_overload = teos_t{overload};
        check_vals(te_overload);

        auto te2 = std::move(te_overload);
        check_vals(te2);

        size_t i    = 0;
        te_overload = [&i, a = a](const auto&) {
            return a[i++];
        };
        check_vals(te_overload);

        te_overload = teos_t{};
        CHECK(!te_overload);
    }
}

TEST_CASE("UniVector", "[util]")
{
    auto vec = util::UniVector< int, std::string >{};
    REQUIRE(vec.size() == 0);

    constexpr size_t n_ints = 50'000, n_strs = 20'000;
    std::fill_n(std::back_inserter(vec.getVector< int >()), n_ints, 42);
    std::fill_n(std::back_inserter(vec.getVector< std::string >()), n_strs, "42");
    REQUIRE(vec.size() == n_ints + n_strs);

    SECTION("Iteration")
    {
        std::mutex            mut; // Catch2 assertions not thread safe
        std::atomic< size_t > n_visited{0};
        const auto            check          = util::OverloadSet{[&](int i) {
                                                 n_visited.fetch_add(1, std::memory_order_relaxed);
                                                 const auto lock = std::lock_guard{mut};
                                                 CHECK(i == 42);
                                             },
                                             [&](const std::string& str) {
                                                 n_visited.fetch_add(1, std::memory_order_relaxed);
                                                 const auto lock = std::lock_guard{mut};
                                                 CHECK(std::stoi(str) == 42);
                                             }};
        const auto            check_n_visits = [&]() {
            REQUIRE(n_visited == n_ints + n_strs);
            n_visited.store(0);
        };

        vec.visit(check, std::execution::seq);
        check_n_visits();
        std::as_const(vec).visit(check, std::execution::seq);
        check_n_visits();

        vec.visit(check, std::execution::par);
        check_n_visits();
        std::as_const(vec).visit(check, std::execution::par);
        check_n_visits();
    }

    SECTION("Reduction")
    {
        const auto transform = util::OverloadSet{[&](int i) { return i; },
                                                 [&](const std::string& str) {
                                                     return std::stoi(str);
                                                 }};

        CHECK(vec.transformReduce(0, transform, std::plus{}, std::execution::seq) == (n_strs + n_ints) * 42);
        CHECK(vec.transformReduce(0, transform, std::plus{}, std::execution::par) == (n_strs + n_ints) * 42);
    }

    SECTION("Lookup")
    {
        auto       find_val = 42;
        const auto pred     = util::OverloadSet{[&find_val](int i) { return i == find_val; },
                                            [&find_val](const std::string& str) {
                                                return std::stoi(str) == find_val;
                                            }};

        {
            const auto found_ptr = vec.find(pred);
            REQUIRE(found_ptr.has_value());
            REQUIRE(*found_ptr == vec.at(0));
        }

        vec.getVector< std::string >().emplace_back("43");
        {
            find_val             = 43;
            const auto found_ptr = vec.find(pred);
            REQUIRE(found_ptr.has_value());
            REQUIRE(*found_ptr == vec.at(n_ints + n_strs));
        }

        find_val = 44;
        REQUIRE_FALSE(vec.find(pred).has_value());
        REQUIRE_FALSE(std::as_const(vec).find(pred).has_value());
    }
}

TEST_CASE("CrsGraph", "[util]")
{
    using VertexType                        = size_t;
    constexpr size_t size                   = 1uz << 7;
    const auto       check_dfs_returns_iota = [&](const util::CrsGraph< VertexType >& graph) {
        auto visited = std::vector< VertexType >{};
        visited.reserve(size);
        util::depthFirstSearch(graph, [&](VertexType v) { visited.push_back(v); });
        CHECK(std::ranges::equal(visited, std::views::iota(VertexType{0}, size)));
    };

    SECTION("Unary tree")
    {
        const auto graph = util::makeCrsGraph(std::views::iota(0uz, size - 1), std::views::iota(1uz, size));
        for (VertexType v = 0; v != size - 1; ++v)
            REQUIRE(graph(v).size() == 1);
        REQUIRE(graph(size - 1).size() == 0);
        check_dfs_returns_iota(graph);
    }
    SECTION("Chain")
    {
        auto from = std::views::iota(0uz, size);
        auto to   = util::ArrayOwner< VertexType >(from);
        std::ranges::rotate(to, std::next(to.begin()));
        const auto graph = util::makeCrsGraph(from, to, util::GraphType::Undirected);
        for (VertexType v = 0; v != size; ++v)
            REQUIRE(graph(v).size() == 2);
        check_dfs_returns_iota(graph);
    }
    SECTION("Complete")
    {
        auto       verts     = std::views::iota(0uz, size);
        auto       all_combs = std::views::cartesian_product(verts, verts);
        const auto graph     = util::makeCrsGraph(all_combs | std::views::keys, all_combs | std::views::values);
        for (VertexType v = 0; v != size; ++v)
            REQUIRE(graph(v).size() == size);
        check_dfs_returns_iota(graph);
    }
    SECTION("Cluster")
    {
        auto from = std::array{std::views::iota(0uz, size / 2), std::views::iota(size / 2, size)};
        auto to   = std::array{util::ArrayOwner(from.front()), util::ArrayOwner(from.back())};
        for (auto& t : to)
            std::ranges::rotate(t, std::next(t.begin()));
        const auto graph = util::makeCrsGraph(from | std::views::join, to | std::views::join);
        for (VertexType v = 0; v != size; ++v)
            REQUIRE(graph(v).size() == 1);
        auto visited = std::vector< VertexType >{};
        visited.reserve(size / 2);
        util::depthFirstSearch(graph, [&](VertexType v) { visited.push_back(v); });
        CHECK(std::ranges::equal(visited, std::views::iota(VertexType{0}, size / 2)));
        visited.clear();
        util::depthFirstSearch(graph, [&](VertexType v) { visited.push_back(v); }, size / 2);
        CHECK(std::ranges::equal(visited, std::views::iota(VertexType{size / 2}, VertexType{size})));
    }
    SECTION("Empty")
    {
        const auto graph = util::makeCrsGraph(std::views::empty< VertexType >, std::views::empty< VertexType >);
        REQUIRE(graph.getNRows() == 0);
    }
}

TEST_CASE("Spatial hash table", "[util]")
{
    constexpr auto format_point = []< Arithmetic_c T, size_t dim >(const std::array< T, dim >& point) {
        auto retval = std::string{"("};
        std::ranges::copy(point | std::views::transform([](auto v) { return std::to_string(v); }) |
                              std::views::join_with(std::string_view{", "}),
                          std::back_inserter(retval));
        retval.push_back(')');
        return retval;
    };
    constexpr int num_ticks = 6;
    auto ticks = std::views::iota(0, num_ticks) | std::views::transform([](int i) { return static_cast< double >(i); });
    auto xyz   = std::views::cartesian_product(ticks, ticks, ticks);

    SECTION("Coarse grid")
    {
        constexpr auto dx_coarse  = 1.;
        auto           space_hash = util::SpatialHashTable< std::string, 3 >{dx_coarse};
        for (const auto& [x, y, z] : xyz)
        {
            const auto point = std::array{x, y, z};
            space_hash.insert(point, format_point(point));
        }
        REQUIRE(space_hash.size() == std::ranges::size(xyz));
        CHECK(static_cast< size_t >(std::ranges::distance(space_hash.all())) == space_hash.size());
        for (const auto& [x, y, z] : xyz)
        {
            const auto point = std::array{x, y, z};
            auto       prox  = space_hash.proximate(point);
            CHECK(std::ranges::contains(prox | std::views::keys, point));
        }
        auto ticks_inner = ticks | std::views::drop(1) | std::views::take(num_ticks - 2);
        auto xyz_inner   = std::views::cartesian_product(ticks_inner, ticks_inner, ticks_inner);
        for (const auto& [x, y, z] : xyz_inner)
        {
            const auto point = std::array{x, y, z};
            auto       prox  = space_hash.proximate(point);
            CHECK(std::ranges::distance(prox) == 27);
        }
    }

    SECTION("Fine grid")
    {
        constexpr auto dx_fine    = .01;
        auto           space_hash = util::SpatialHashTable< std::string, 3 >{dx_fine};
        for (const auto& [x, y, z] : xyz)
        {
            const auto point = std::array{x, y, z};
            space_hash.insert(point, format_point(point));
        }
        REQUIRE(space_hash.size() == std::ranges::size(xyz));
        CHECK(static_cast< size_t >(std::ranges::distance(space_hash.all())) == space_hash.size());
        for (const auto& [x, y, z] : xyz)
        {
            const auto point = std::array{x, y, z};
            auto       prox  = space_hash.proximate(point);
            CHECK(std::ranges::contains(prox | std::views::keys, point));
            CHECK(std::ranges::distance(prox) == 1);
        }
    }
}
