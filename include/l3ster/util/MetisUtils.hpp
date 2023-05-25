#ifndef L3STER_UTIL_METISUTILS_HPP
#define L3STER_UTIL_METISUTILS_HPP

#include "l3ster/util/Assertion.hpp"

extern "C"
{
#include "metis.h"
}

#include <memory>
#include <span>

namespace lstr::util::metis
{
class GraphWrapper
{
    struct Deleter // defining explicitly, gcc warns about anonymous namespaces when using a lambda
    {
        void operator()(idx_t* ptr) const noexcept { METIS_Free(ptr); }
    };
    using array_t = std::unique_ptr< idx_t[], Deleter >; // NOLINT

public:
    using span_t = std::span< const idx_t >;

    inline GraphWrapper(const GraphWrapper& other);
    GraphWrapper(GraphWrapper&&) noexcept = default;
    inline GraphWrapper& operator=(const GraphWrapper& other);
    GraphWrapper&        operator=(GraphWrapper&&) noexcept = default;
    ~GraphWrapper()                                         = default;

    GraphWrapper(idx_t* xa, idx_t* adj, size_t nv) : xadj{xa}, adjncy{adj}, nvert{nv} {}

    [[nodiscard]] span_t getXadj() const { return {xadj.get(), nvert + 1}; }
    [[nodiscard]] span_t getAdjncy() const { return {adjncy.get(), static_cast< size_t >(getXadj().back())}; }

    [[nodiscard]] inline span_t getElementAdjacent(size_t el_id) const;

private:
    array_t xadj, adjncy;
    size_t  nvert;
};

GraphWrapper::GraphWrapper(const GraphWrapper& other)
    : xadj{static_cast< idx_t* >(malloc((other.getXadj().size()) * sizeof(idx_t)))},     // NOLINT
      adjncy{static_cast< idx_t* >(malloc((other.getAdjncy().size()) * sizeof(idx_t)))}, // NOLINT
      nvert{other.nvert}
{
    std::ranges::copy(other.getXadj(), xadj.get());
    std::ranges::copy(other.getAdjncy(), adjncy.get());
}

GraphWrapper& GraphWrapper::operator=(const GraphWrapper& other)
{
    if (this == &other)
        return *this;

    xadj.reset(static_cast< idx_t* >(malloc((other.getXadj().size()) * sizeof(idx_t))));     // NOLINT
    adjncy.reset(static_cast< idx_t* >(malloc((other.getAdjncy().size()) * sizeof(idx_t)))); // NOLINT
    nvert = other.nvert;
    std::ranges::copy(other.getXadj(), xadj.get());
    std::ranges::copy(other.getAdjncy(), adjncy.get());
    return *this;
}

GraphWrapper::span_t GraphWrapper::getElementAdjacent(size_t el_id) const
{
    const auto adjncy_begin_ind = getXadj()[el_id];
    const auto adjncy_end_ind   = getXadj()[el_id + 1];
    const auto adjncy_size      = adjncy_end_ind - adjncy_begin_ind;
    return getAdjncy().subspan(adjncy_begin_ind, adjncy_size);
}

inline void handleMetisErrorCode(int err_code, std::source_location src_loc = std::source_location::current())
{
    util::throwingAssert< std::bad_alloc >(err_code != METIS_ERROR_MEMORY, {}, src_loc);
    util::throwingAssert(err_code == METIS_OK, "Metis runtime error", src_loc);
}
} // namespace lstr::util::metis
#endif // L3STER_UTIL_METISUTILS_HPP
