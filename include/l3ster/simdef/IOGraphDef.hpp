#ifndef L3STER_SIMSTRUCTURE_IOGRAPHDEF_HPP
#define L3STER_SIMSTRUCTURE_IOGRAPHDEF_HPP

#include "FieldDef.hpp"
#include "util/ConstexprVector.hpp"

#include <algorithm>
#include <tuple>

namespace lstr::def
{
enum struct ViewTypes
{
    Constant,
    Static,
    CurrentIter,
    Snapshot
};

template < ViewTypes VT >
struct FieldView
{
    template < Space S, Time T >
    constexpr FieldView(Field< S, T >& field_) : field{static_cast< void* >(&field_)}
    {}

    void* field;
};

class IOGraph
{
public:
    constexpr IOGraph()
    {
        constexpr size_t init_size{4u};
        graph.reserve(init_size);
    }

    template < ViewTypes VT, typename Kernel >
    constexpr void define(const FieldView< VT >& view, Kernel& kernel)
    {
        graph.emplaceBack(view.field, static_cast< void* >(&kernel), VT);
    }

    [[nodiscard]] constexpr const auto& getGraph() const { return graph; }

    constexpr void finalize()
    {
        std::ranges::sort(graph, [](const auto& t1, const auto& t2) {
            return std::tie(get< 0 >(t1), get< 1 >(t1)) < std::tie(get< 0 >(t2), get< 1 >(t2));
        });
    }

private:
    ConstexprVector< std::tuple< void*, void*, ViewTypes > > graph{};
};
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_IOGRAPHDEF_HPP
