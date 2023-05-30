#ifndef L3STER_MESH_ELEMENTINTERSECTING_HPP
#define L3STER_MESH_ELEMENTINTERSECTING_HPP

#include "l3ster/mesh/ElementTypes.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Meta.hpp"

#include <bitset>
#include <span>

namespace lstr
{
namespace detail
{
template < ElementTypes T, el_o_t O, el_side_t S1, el_side_t S2 >
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

template < ElementTypes T, el_o_t O, el_side_t S1, el_side_t S2 >
consteval auto getSideIntersection()
{
    constexpr auto intersect_raw = intersectElementSides< T, O, S1, S2 >();
    return util::trimArray< intersect_raw.second >(intersect_raw.first);
}

template < tuple_like T, size_t N >
struct tuple_largerequal : public std::conditional_t< std::tuple_size_v< T > >= N, std::true_type, std::false_type >
{};

template < tuple_like T >
using tuple_le2 = tuple_largerequal< T, 2 >;

template < ElementTypes T, el_o_t O, el_side_t S, el_side_t... SIDES >
consteval auto intersectSideWith(std::integer_sequence< el_side_t, SIDES... >)
{
    return util::makeTupleIf< tuple_le2 >(getSideIntersection< T, O, S, SIDES >()...);
}

template < ElementTypes T, el_o_t O, el_side_t S >
consteval auto intersectSideWithSubseqSides()
{
    constexpr el_side_t max_side_ind = ElementTraits< Element< T, O > >::n_sides - 1;
    if constexpr (S < max_side_ind)
        return intersectSideWith< T, O, S >(util::int_seq_interval< el_side_t, S + el_side_t{1}, max_side_ind >{});
    else
        return std::tuple<>{};
}

template < ElementTypes T, el_o_t O, el_side_t... SIDES >
consteval auto intersectSidesWithSubseqSides(std::integer_sequence< el_side_t, SIDES... >)
{
    return std::tuple_cat(intersectSideWithSubseqSides< T, O, SIDES >()...);
}

template < ElementTypes T, el_o_t O >
consteval auto makeElementEdgeTable()
{
    if constexpr (Element< T, O >::native_dim < 3)
        return std::tuple<>{};
    else
        return intersectSidesWithSubseqSides< T, O >(
            std::make_integer_sequence< el_side_t, ElementTraits< Element< T, O > >::n_sides - 1 >{});
}

template < ElementTypes T, el_o_t O >
consteval auto getElementIndices()
{
    return std::make_tuple(util::consecutiveIndices(std::integral_constant< el_locind_t, Element< T, O >::n_nodes >{}));
}

template < ElementTypes T, el_o_t O, dim_t DIM >
consteval auto getElementOuterFeatures()
    requires(DIM <= 3)
{
    using el_traits_t     = ElementTraits< Element< T, O > >;
    constexpr auto el_dim = el_traits_t::native_dim;

    if constexpr (DIM == 0 or el_dim < DIM)    // 3/9 cases: (1, 2), (1, 3), (2, 3)
        return std::tuple<>{};                 //
    else if constexpr (el_dim == DIM)          // 3/9 cases: (1, 1), (2, 2), (3, 3)
        return getElementIndices< T, O >();    //
    else if constexpr (DIM == el_dim - 1)      // 2/9 cases: (2, 1), (3, 2)
        return el_traits_t::boundary_table;    //
    else                                       // 1/9 cases: (3, 1)
        return makeElementEdgeTable< T, O >(); //
}
} // namespace detail

template < ElementTypes T, el_o_t O, dim_t DIM >
constexpr inline auto element_outer_features = detail::getElementOuterFeatures< T, O, DIM >();

namespace detail
{
template < ElementTypes T1, ElementTypes T2 >
consteval dim_t getHighestMatchableDim()
{
    constexpr auto d1        = Element< T1, 1 >::native_dim;
    constexpr auto d2        = Element< T2, 1 >::native_dim;
    constexpr auto lower_dim = d1 <= d2 ? d1 : d2;
    return d1 == d2 ? d1 - 1 : lower_dim;
}

template < dim_t DIM, el_o_t O, ElementTypes T1, ElementTypes T2 >
auto elementIntersectionSpansAtDim(const Element< T1, 1 >& el1, const Element< T2, 1 >& el2)
{
    using span_t = std::span< const el_locind_t >;
    span_t           intersect_inds1, intersect_inds2;
    const auto&      f1_o1   = element_outer_features< T1, 1, DIM >;
    const auto&      f2_o1   = element_outer_features< T2, 1, DIM >;
    const auto&      f1_oN   = element_outer_features< T1, O, DIM >;
    const auto&      f2_oN   = element_outer_features< T2, O, DIM >;
    constexpr size_t f1_size = std::tuple_size_v< std::decay_t< decltype(f1_o1) > >;
    constexpr size_t f2_size = std::tuple_size_v< std::decay_t< decltype(f2_o1) > >;

    std::invoke(
        [&]< size_t... I1 >(std::index_sequence< I1... >) {
            const auto iterate_over_f1 = [&]< size_t Ind1 >(std::integral_constant< size_t, Ind1 >) {
                auto nodes1 = util::arrayAtInds(el1.getNodes(), std::get< Ind1 >(f1_o1));
                std::ranges::sort(nodes1);
                return std::invoke(
                    [&]< size_t... I2 >(std::index_sequence< I2... >) {
                        const auto iterate_over_f2 = [&]< size_t Ind2 >(std::integral_constant< size_t, Ind2 >) {
                            auto nodes2 = util::arrayAtInds(el2.getNodes(), std::get< Ind2 >(f2_o1));
                            std::ranges::sort(nodes2);
                            const bool is_matched = std::ranges::equal(nodes1, nodes2);
                            if (is_matched)
                            {
                                intersect_inds1 = span_t{std::get< Ind1 >(f1_oN)};
                                intersect_inds2 = span_t{std::get< Ind2 >(f2_oN)};
                            }
                            return is_matched;
                        };
                        return (iterate_over_f2(std::integral_constant< size_t, I2 >{}) or ...);
                    },
                    std::make_index_sequence< f2_size >{});
            };
            (iterate_over_f1(std::integral_constant< size_t, I1 >{}) or ...);
        },
        std::make_index_sequence< f1_size >{});

    return std::make_pair(intersect_inds1, intersect_inds2);
}

template < el_o_t O, ElementTypes T1, ElementTypes T2 >
std::pair< std::span< const el_locind_t >, std::span< const el_locind_t > >
elementIntersection(const Element< T1, 1 >& el1, const Element< T2, 1 >& el2)
{
    constexpr auto highest_matchable = detail::getHighestMatchableDim< T1, T2 >();

    if constexpr (highest_matchable == 0) // metis marks 2 lines as neighbours in the dual graph
        return {};
    else
    {
        if constexpr (highest_matchable == 1)
            return detail::elementIntersectionSpansAtDim< 1, O >(el1, el2);
        else // highest_matchable == 2
        {
            const auto match_dim2 = detail::elementIntersectionSpansAtDim< 2, O >(el1, el2);
            if (not match_dim2.first.empty())
                return match_dim2;
            else
                return detail::elementIntersectionSpansAtDim< 1, O >(el1, el2);
        }
    }
}

consteval auto getPointMatcher()
{
    return [](const Point< 3 >& p1, const Point< 3 >& p2) {
        constexpr double tolerance = 1e-12;
        return (Eigen::Vector< val_t, 3 >{p1} - Eigen::Vector< val_t, 3 >{p2}).norm() < tolerance;
    };
}
} // namespace detail

template < el_o_t O, ElementTypes T >
constexpr void updateMatchMask(const Element< T, 1 >&                   el_o1,
                               std::bitset< Element< T, O >::n_nodes >& mask,
                               typename Element< T, O >::node_array_t&  nodes)
{
    const auto el_o1_nodelocs = nodePhysicalLocation(el_o1);
    const auto el_oN_nodelocs = nodePhysicalLocation(Element< T, O >{{}, el_o1.getData(), 0});
    std::array< size_t, std::tuple_size_v< decltype(el_o1_nodelocs) > > matched_inds;
    util::matchingPermutation(el_oN_nodelocs, el_o1_nodelocs, begin(matched_inds), detail::getPointMatcher());
    for (size_t i = 0; auto ind : matched_inds)
    {
        mask[ind]  = true;
        nodes[ind] = el_o1.getNodes()[i++];
    }
}

template < ElementTypes T_converted, ElementTypes T_converting, el_o_t O >
constexpr void updateMatchMask(const Element< T_converted, 1 >&                    pattern_o1_el,
                               const Element< T_converted, O >&                    pattern_oN_el,
                               const Element< T_converting, 1 >&                   match_o1_el,
                               std::bitset< Element< T_converting, O >::n_nodes >& mask,
                               typename Element< T_converting, O >::node_array_t&  nodes)
{
    const auto [pattern_inds, match_inds] = detail::elementIntersection< O >(pattern_o1_el, match_o1_el);
    std::array< size_t, std::tuple_size_v< std::decay_t< decltype(nodes) > > > match_buffer;
    const Element< T_converting, O > match_oN_el{{}, match_o1_el.getData(), {}};
    auto filtered_match_inds = match_inds | std::views::filter([&](auto ind) { return not mask[ind]; });

    util::matchingPermutation(
        pattern_inds | std::views::transform([&](auto ind) { return nodePhysicalLocation(pattern_oN_el, ind); }),
        filtered_match_inds | std::views::transform([&](auto ind) { return nodePhysicalLocation(match_oN_el, ind); }),
        begin(match_buffer),
        detail::getPointMatcher());

    for (size_t i = 0u; auto m_ind : filtered_match_inds)
    {
        mask[m_ind]  = true;
        nodes[m_ind] = pattern_oN_el.getNodes()[pattern_inds[match_buffer[i++]]];
    }
}
} // namespace lstr
#endif // L3STER_MESH_ELEMENTINTERSECTING_HPP
