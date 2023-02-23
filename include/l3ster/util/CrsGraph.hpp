#ifndef L3STER_UTIL_CRSGRAPH_HPP
#define L3STER_UTIL_CRSGRAPH_HPP

#include <concepts>
#include <execution>
#include <iterator>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>

namespace lstr::util
{
template < std::integral VertexType >
class CrsGraph
{
public:
    CrsGraph() = default;
    template < std::ranges::range R >
        requires std::convertible_to< std::ranges::range_value_t< R >, std::size_t >
    CrsGraph(R&& adj_sizes);

    auto operator()(std::size_t vertex_ind) noexcept -> std::span< VertexType >;
    auto operator()(std::size_t vertex_ind) const noexcept -> std::span< const VertexType >;
    auto size() const noexcept -> std::size_t { return m_adj_offsets.size() - 1; }

private:
    auto initAdjOffsets(auto&& adj_sizes) -> ArrayOwner< std::size_t >;

    ArrayOwner< std::size_t > m_adj_offsets;
    ArrayOwner< VertexType >  m_adjacent;
};

template < std::integral VertexType >
auto CrsGraph< VertexType >::operator()(std::size_t vertex_ind) noexcept -> std::span< VertexType >
{
    return {std::next(m_adjacent.begin(), m_adj_offsets[vertex_ind]),
            std::next(m_adjacent.begin(), m_adj_offsets[vertex_ind + 1u])};
}

template < std::integral VertexType >
auto CrsGraph< VertexType >::operator()(std::size_t vertex_ind) const noexcept -> std::span< const VertexType >
{
    return {std::next(m_adjacent.get(), m_adj_offsets[vertex_ind]),
            std::next(m_adjacent.get(), m_adj_offsets[vertex_ind + 1u])};
}

template < std::integral VertexType >
auto CrsGraph< VertexType >::initAdjOffsets(auto&& adj_sizes) -> ArrayOwner< std::size_t >
{
    auto retval           = ArrayOwner< std::size_t >(std::ranges::distance(adj_sizes) + 1);
    retval[0]             = 0;
    auto adj_sizes_common = std::views::common(std::forward< decltype(adj_sizes) >(adj_sizes));
    std::inclusive_scan(std::execution::par,
                        std::ranges::cbegin(adj_sizes_common),
                        std::ranges::cend(adj_sizes_common),
                        std::next(retval.begin()));
    return retval;
}

template < std::integral VertexType >
template < std::ranges::range R >
    requires std::convertible_to< std::ranges::range_value_t< R >, std::size_t >
CrsGraph< VertexType >::CrsGraph(R&& adj_sizes)
    : m_adj_offsets{initAdjOffsets(std::forward< decltype(adj_sizes) >(adj_sizes))}, m_adjacent{m_adj_offsets.back()}
{}
} // namespace lstr::util
#endif // L3STER_UTIL_CRSGRAPH_HPP
