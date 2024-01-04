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
    explicit CrsGraph(R&& adj_sizes)
        requires std::convertible_to< std::ranges::range_value_t< R >, std::size_t >;

    auto operator()(std::size_t vertex_ind) noexcept -> std::span< VertexType >;
    auto operator()(std::size_t vertex_ind) const noexcept -> std::span< const VertexType >;
    auto getNRows() const noexcept -> std::size_t { return m_adj_offsets.size() - 1; }
    auto getNEntries() const noexcept -> std::size_t { return m_adjacent.size(); }

private:
    template < typename R >
    static auto initAdjOffsets(R&& adj_sizes) -> ArrayOwner< std::size_t >;

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
    return {std::next(m_adjacent.begin(), m_adj_offsets[vertex_ind]),
            std::next(m_adjacent.begin(), m_adj_offsets[vertex_ind + 1u])};
}

template < std::integral VertexType >
template < typename R >
auto CrsGraph< VertexType >::initAdjOffsets(R&& adj_sizes) -> ArrayOwner< std::size_t >
{
    auto retval           = ArrayOwner< std::size_t >(std::ranges::distance(adj_sizes) + 1);
    retval[0]             = 0;
    auto adj_sizes_common = std::forward< R >(adj_sizes) | std::views::common;
    std::inclusive_scan(std::execution::par,
                        std::ranges::cbegin(adj_sizes_common),
                        std::ranges::cend(adj_sizes_common),
                        std::next(retval.begin()));
    return retval;
}

template < std::integral VertexType >
template < std::ranges::range R >
CrsGraph< VertexType >::CrsGraph(R&& adj_sizes)
    requires std::convertible_to< std::ranges::range_value_t< R >, std::size_t >
    : m_adj_offsets(initAdjOffsets(std::forward< R >(adj_sizes))), m_adjacent(m_adj_offsets.back())
{}
} // namespace lstr::util
#endif // L3STER_UTIL_CRSGRAPH_HPP
