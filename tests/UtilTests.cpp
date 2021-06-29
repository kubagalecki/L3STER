#include "util/ConstexprVector.hpp"

#include "catch2/catch.hpp"

#include <algorithm>

static consteval auto getConstexprVecParams(int size, int cap)
{
    lstr::ConstexprVector< int > v;
    for (int i = 0; i < cap; ++i)
        v.pushBack(i);
    for (int i = cap; i > size; --i)
        v.popBack();
    return std::make_pair(v.size(), v.capacity());
}

static consteval bool checkConstexprVecStorage()
{
    lstr::ConstexprVector< lstr::ConstexprVector< int > > v(3, lstr::ConstexprVector< int >{0, 1, 2});
    bool                                                  ret = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ret &= v[i][j] == j;
    return ret;
}

static consteval auto checkConstexprVectorReserve(int cap)
{
    lstr::ConstexprVector< int > v;
    v.reserve(cap);
    return std::make_pair(v.size(), v.capacity());
}

static consteval bool checkConstexprVectorIters()
{
    lstr::ConstexprVector< int > v;
    constexpr int                size = 10;
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
