#ifndef L3STER_UTIL_CRSGRAPH_HPP
#define L3STER_UTIL_CRSGRAPH_HPP

#include <concepts>
#include <iterator>
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

    auto operator()(std::size_t vertex_ind) -> std::span< VertexType >;
    auto operator()(std::size_t vertex_ind) const -> std::span< const VertexType >;
    auto operator()(std::size_t begin, std::size_t end) -> std::span< VertexType >;
    auto operator()(std::size_t begin, std::size_t end) const -> std::span< const VertexType >;
    auto getNRows() const -> std::size_t { return m_adj_offsets.size() - 1; }
    auto getNEntries() const -> std::size_t { return m_adjacent.size(); }
    auto getRawEntries() const -> std::span< const VertexType > { return {m_adjacent}; }
    auto data() -> VertexType* { return m_adjacent.data(); }
    auto data() const -> const VertexType* { return m_adjacent.data(); }

private:
    template < typename R >
    static auto initAdjOffsets(R&& adj_sizes) -> ArrayOwner< std::ptrdiff_t >;

    ArrayOwner< std::ptrdiff_t > m_adj_offsets{0};
    ArrayOwner< VertexType >     m_adjacent{};
};

template < std::integral VertexType >
auto CrsGraph< VertexType >::operator()(std::size_t vertex_ind) -> std::span< VertexType >
{
    return {std::next(m_adjacent.begin(), m_adj_offsets.at(vertex_ind)),
            std::next(m_adjacent.begin(), m_adj_offsets.at(vertex_ind + 1u))};
}

template < std::integral VertexType >
auto CrsGraph< VertexType >::operator()(std::size_t vertex_ind) const -> std::span< const VertexType >
{
    return {std::next(m_adjacent.begin(), m_adj_offsets.at(vertex_ind)),
            std::next(m_adjacent.begin(), m_adj_offsets.at(vertex_ind + 1u))};
}

template < std::integral VertexType >
auto CrsGraph< VertexType >::operator()(std::size_t begin, std::size_t end) -> std::span< VertexType >
{
    return {std::next(m_adjacent.begin(), m_adj_offsets.at(begin)),
            std::next(m_adjacent.begin(), m_adj_offsets.at(end))};
}

template < std::integral VertexType >
auto CrsGraph< VertexType >::operator()(std::size_t begin, std::size_t end) const -> std::span< const VertexType >
{
    return {std::next(m_adjacent.begin(), m_adj_offsets.at(begin)),
            std::next(m_adjacent.begin(), m_adj_offsets.at(end))};
}

template < std::integral VertexType >
template < typename R >
auto CrsGraph< VertexType >::initAdjOffsets(R&& adj_sizes) -> ArrayOwner< std::ptrdiff_t >
{
    auto retval           = ArrayOwner< std::ptrdiff_t >(static_cast< size_t >(std::ranges::distance(adj_sizes) + 1));
    retval.front()        = 0;
    auto adj_sizes_common = std::forward< R >(adj_sizes) | std::views::common;
    std::inclusive_scan(adj_sizes_common.begin(), adj_sizes_common.end(), std::next(retval.begin()));
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
