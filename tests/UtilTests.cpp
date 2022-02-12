#include "l3ster/util/Common.hpp"
#include "l3ster/util/ConstexprVector.hpp"
#include "l3ster/util/Meta.hpp"
#include "l3ster/util/MetisUtils.hpp"
#include "l3ster/util/SetStackSize.hpp"

#include "catch2/catch.hpp"

#include <random>

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

TEST_CASE("Constexpr vector test", "[util]")
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

TEST_CASE("Stack size manipulation test", "[util]")
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

TEMPLATE_TEST_CASE("Bitset (de-)serialization tests",
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

TEST_CASE("MetisGraphWrapper tests", "[util]")
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
