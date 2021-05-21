#ifndef L3STER_UTIL_METISUTILS_HPP
#define L3STER_UTIL_METISUTILS_HPP
#include "metis.h"

#include <memory>
#include <span>

namespace lstr
{
class MetisGraphWrapper
{
    struct Deleter // defining explicitly, gcc warns about anonymous namespaces when using a lambda
    {
        void operator()(idx_t* ptr) const noexcept { METIS_Free(ptr); }
    };
    using array_t = std::unique_ptr< idx_t[], Deleter >;

public:
    using span_t = std::span< const idx_t >;

    MetisGraphWrapper(idx_t* xa, idx_t* adj, size_t nv) : xadj{xa}, adjncy{adj}, nvert{nv} {}

    [[nodiscard]] span_t getXadj() const { return {xadj.get(), nvert + 1}; }
    [[nodiscard]] span_t getAdjncy() const { return {adjncy.get(), static_cast< size_t >(getXadj().back())}; }

    [[nodiscard]] inline span_t getElementAdjacent(size_t el_id) const;

private:
    array_t xadj, adjncy;
    size_t  nvert;
};

MetisGraphWrapper::span_t MetisGraphWrapper::getElementAdjacent(size_t el_id) const
{
    const auto adjncy_begin_ind = getXadj()[el_id];
    const auto adjncy_end_ind   = getXadj()[el_id + 1];
    const auto adjncy_size      = adjncy_end_ind - adjncy_begin_ind;
    return getAdjncy().subspan(adjncy_begin_ind, adjncy_size);
}

namespace detail
{
inline void handleMetisErrorCode(int error)
{
    switch (error)
    {
    case METIS_OK:
        break;
    case METIS_ERROR_MEMORY:
        throw std::bad_alloc{};
    default:
        throw std::runtime_error{"Metis failed to partition the mesh"};
    }
}

} // namespace detail
} // namespace lstr
#endif // L3STER_UTIL_METISUTILS_HPP
