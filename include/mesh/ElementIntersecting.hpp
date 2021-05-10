#ifndef L3STER_MESH_ELEMENTINTERSECTING_HPP
#define L3STER_MESH_ELEMENTINTERSECTING_HPP
#include "mesh/ElementTypes.hpp"
#include "util/Algorithm.hpp"

#include <bitset>

namespace lstr
{
namespace detail
{
template < ElementTypes T, el_o_t O, el_ns_t S1, el_ns_t S2 >
consteval auto intersectElementSides()
{
    auto face1 = std::get< S1 >(ElementTraits< Element< T, O > >::boundary_table);
    auto face2 = std::get< S2 >(ElementTraits< Element< T, O > >::boundary_table);
    std::ranges::sort(face1);
    std::ranges::sort(face2);
    constexpr size_t min_face_size =
        std::min({std::tuple_size_v< decltype(face1) >, std::tuple_size_v< decltype(face2) >});
    using value_type = typename decltype(face1)::value_type;
    std::array< value_type, min_face_size > intersection_buf;
    const auto it = std::set_intersection(begin(face1), end(face1), begin(face2), end(face2), begin(intersection_buf));
    const size_t intersection_size = std::distance(begin(intersection_buf), it);
    return std::make_pair(intersection_buf, intersection_size);
}

template < ElementTypes T, el_o_t O, el_ns_t S1, el_ns_t S2 >
consteval auto getSideIntersection()
{
    constexpr auto intersect_raw = intersectElementSides< T, O, S1, S2 >();
    return trimArray< intersect_raw.second >(intersect_raw.first);
}

template < tuple_like T, size_t N >
struct tuple_largerequal : public std::conditional_t< std::tuple_size_v< T > >= N, std::true_type, std::false_type >
{};

template < tuple_like T >
using tuple_le2 = tuple_largerequal< T, 2 >;

template < ElementTypes T, el_o_t O, el_ns_t S, el_ns_t... SIDES >
consteval auto intersectSideWith(std::integer_sequence< el_ns_t, SIDES... >)
{
    return makeTupleIf< tuple_le2 >(getSideIntersection< T, O, S, SIDES >()...);
}

template < ElementTypes T, el_o_t O, el_ns_t S >
consteval auto intersectSideWithSubseqSides()
{
    constexpr el_ns_t max_side_ind = ElementTraits< Element< T, O > >::n_sides - 1;
    if constexpr (S < max_side_ind)
        return intersectSideWith< T, O, S >(detail::int_seq_interval< el_ns_t, S + el_ns_t{1}, max_side_ind >{});
    else
        return std::tuple<>{};
}

template < ElementTypes T, el_o_t O, el_ns_t... SIDES >
consteval auto intersectSidesWithSubseqSides(std::integer_sequence< el_ns_t, SIDES... >)
{
    return std::tuple_cat(intersectSideWithSubseqSides< T, O, SIDES >()...);
}

template < ElementTypes T, el_o_t O >
consteval auto makeElementEdgeTable()
{
    if constexpr (ElementTraits< Element< T, O > >::native_dim < 3)
        return std::tuple<>{};
    else
        return intersectSidesWithSubseqSides< T, O >(
            std::make_integer_sequence< el_ns_t, ElementTraits< Element< T, O > >::n_sides - 1 >{});
}

template < ElementTypes T, el_o_t O >
consteval auto getElementIndices()
{
    return std::make_tuple(consecutiveIndices(std::integral_constant< el_locind_t, Element< T, O >::n_nodes >{}));
}

template < ElementTypes T, el_o_t O, dim_t DIM >
requires(DIM <= 3) consteval auto getElementOuterFeatures()
{
    using el_traits_t     = ElementTraits< Element< T, O > >;
    constexpr auto el_dim = el_traits_t::native_dim;

    if constexpr (DIM == 0 or el_dim < DIM)    // 3/9 cases: (1, 2), (1, 3), (2, 3)
        return std::tuple<>{};                 //
    else if constexpr (el_dim == DIM)          // 3/9 cases: (1, 1), (2, 2), (3, 3)
        return getElementIndices< T, O >();    //
    else if constexpr (DIM == el_dim - 1)      // 2/9 cases: (2, 1), (3, 2)
        return el_traits_t::boundary_table;    //
    else                                       //
        return makeElementEdgeTable< T, O >(); // 1/9 cases: (3, 1)
}
} // namespace detail

template < ElementTypes T, el_o_t O, dim_t DIM >
constexpr inline auto element_outer_features = detail::getElementOuterFeatures< T, O, DIM >();

namespace detail
{
template < ElementTypes T1, ElementTypes T2 >
consteval dim_t highestMatchableDim()
{
    constexpr auto d1        = ElementTraits< Element< T1, 1 > >::native_dim;
    constexpr auto d2        = ElementTraits< Element< T2, 1 > >::native_dim;
    constexpr auto lower_dim = d1 <= d2 ? d1 : d2;
    return d1 == d2 ? d1 - 1 : lower_dim;
}

template < dim_t DIM, ElementTypes T1, ElementTypes T2 >
constexpr auto elementIntersectionAtDim(const Element< T1, 1 >& e1, const Element< T2, 1 >& e2)
{
    using span_t = std::span< const el_locind_t >;
    span_t      m1, m2;
    const auto& f1    = element_outer_features< T1, 1, DIM >;
    const auto& f2    = element_outer_features< T2, 1, DIM >;
    const bool  found = anyInTuple(
        f1,
        [&](const array auto& inds1) {
            auto nodes1 = arrayAtInds(e1.getNodes(), inds1);
            std::ranges::sort(nodes1);
            return anyInTuple(
                f2,
                [&](const array auto inds2) {
                    auto nodes2 = arrayAtInds(e2.getNodes(), inds2);
                    std::ranges::sort(nodes2);
                    return std::ranges::equal(nodes1, nodes2);
                },
                [&](const array auto inds2) { m2 = inds2; });
        },
        [&](const array auto inds1) { m1 = inds1; });
    return std::make_pair(m1, m2);
}
} // namespace detail

template < ElementTypes T1, ElementTypes T2 >
constexpr std::tuple< dim_t, std::span< const el_locind_t >, std::span< const el_locind_t > >
elementIntersection(const Element< T1, 1 >& e1, const Element< T2, 1 >& e2)
{
    constexpr auto highest_matchable = detail::highestMatchableDim< T1, T2 >();

    if constexpr (highest_matchable == 0) // needs to be instantiated, but will never be called (line + line)
        throw std::logic_error{"Cannot match 2 line elements, mesh topology is incorrect"};
    else
    {
        if constexpr (highest_matchable == 1)
            return std::tuple_cat(std::tuple{highest_matchable}, detail::elementIntersectionAtDim< 1 >(e1, e2));

        // highest_matchable == 2
        const auto [m1, m2] = detail::elementIntersectionAtDim< highest_matchable >(e1, e2);
        if (not m1.empty())
            return std::make_tuple(highest_matchable, m1, m2);
        else
            return std::tuple_cat(std::tuple{dim_t{1}}, detail::elementIntersectionAtDim< 1 >(e1, e2));
    }
}

/*
namespace detail
{
template < ElementTypes T1, ElementTypes T2, el_o_t O, dim_t DIM >
consteval size_t getNPossibleIntersections()
{
    if constexpr (DIM == 0)
        return 0;
    else
    {
        constexpr size_t size1 = std::tuple_size_v< decltype(element_outer_features< T1, O, DIM >) >;
        constexpr size_t size2 = std::tuple_size_v< decltype(element_outer_features< T2, O, DIM >) >;
        return size1 * size2 + getNPossibleIntersections< T1, T2, DIM - 1 >();
    }
}

template < ElementTypes T1, ElementTypes T2, el_o_t O >
consteval auto makeIntersectionTable()
{
    constexpr auto d1                = ElementTraits< Element< T1, O > >::native_dim;
    constexpr auto d2                = ElementTraits< Element< T2, O > >::native_dim;
    constexpr auto lower_dim         = d1 <= d2 ? d1 : d2;
    constexpr auto highest_matchable = d1 == d2 ? d1 - 1 : lower_dim;

    using span_t = std::span< const el_locind_t >;
    std::array< std::pair< span_t, span_t >, getNPossibleIntersections< T1, T2, O, highest_matchable >() > itable;
    size_t                                                                                                 index = 0;
    [&]< size_t... I >(std::index_sequence< I... >)
    {
        const auto fill_dimension = [&]< size_t Ind >(std::integral_constant< size_t, Ind >) {
            if constexpr (Ind == 0)
                return;

            constexpr auto& features1 = element_outer_features< T1, O, Ind >;
            constexpr auto& features2 = element_outer_features< T2, O, Ind >;
            forEachTuple(
                [&](const array auto& f1) {
                    forEachTuple(
                        [&](const array auto& f2) {
                            itable[index] = std::make_pair(span_t{f1}, span_t{f2});
                            ++index; // index increment on separate line for better clang-format indentation
                        },
                        features2);
                },
                features1);
        };
        (..., fill_dimension(std::integral_constant< size_t, I >{}));
    }
    (std::make_index_sequence< highest_matchable + 1 >{});
}
} // namespace detail

template < ElementTypes T1, ElementTypes T2, el_o_t O >
inline constexpr auto intersection_table = detail::makeIntersectionTable< T1, T2, O >();

// template < ElementTypes T1, ElementTypes T2, el_o_t O >
// size_t matchElements(const Element< T1, O >& el1, const Element< T2, O >& el2)
//{
//     // TODO
// }
*/

template < ElementTypes T_converted, ElementTypes T_converting, el_o_t O >
constexpr void updateMatchMask(const Element< T_converted, 1 >&                    converted_o1,
                               const Element< T_converted, O >&                    converted_oN,
                               const Element< T_converting, 1 >&                   converting,
                               std::bitset< Element< T_converting, O >::n_nodes >& mask,
                               typename Element< T_converting, O >::node_array_t&  nodes)
{
    const auto [md, m1, m2] = elementIntersection(converted_o1, converting);
}
} // namespace lstr
#endif // L3STER_MESH_ELEMENTINTERSECTING_HPP
