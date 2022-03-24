#ifndef L3STER_UTIL_DYNAMICBITSET_HPP
#define L3STER_UTIL_DYNAMICBITSET_HPP

#include <atomic>
#include <bitset>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace lstr
{
class DynamicBitset
{
    using ull                             = unsigned long long;
    static constexpr std::size_t ull_bits = sizeof(ull) * 8;

    static inline std::size_t    countBits(ull n) { return std::bitset< ull_bits >{n}.count(); }
    static inline std::ptrdiff_t getEntryInd(std::size_t pos) { return static_cast< std::ptrdiff_t >(pos / ull_bits); }
    static inline std::uint8_t   getBitPos(std::size_t pos) { return pos % ull_bits; }
    static inline ull            getMask(std::uint8_t bit_pos) { return 0b1ull << bit_pos; }
    static inline auto           posInds(std::size_t pos) { return std::make_pair(getEntryInd(pos), getBitPos(pos)); }
    static inline std::size_t    computeAllocUlls(std::size_t size)
    {
        return size % ull_bits == 0 ? size / ull_bits : size / ull_bits + 1;
    }

public:
    DynamicBitset() = default;
    DynamicBitset(std::size_t size_) : m_data(computeAllocUlls(size_), 0ull), m_size(size_) {}
    void resize(std::size_t new_size)
    {
        m_data.resize(computeAllocUlls(new_size));
        m_size = new_size;
    }

    [[nodiscard]] std::size_t count() const noexcept
    {
        return std::transform_reduce(
            begin(m_data), end(m_data), std::size_t{0}, std::plus<>{}, [](AlignedUll v) { return countBits(v.value); });
    }
    [[nodiscard]] std::size_t size() const noexcept { return m_size; }

    [[nodiscard]] bool test(std::size_t pos) const noexcept
    {
        const auto [ull_ind, bit_pos] = posInds(pos);
        return m_data[ull_ind].value & getMask(bit_pos);
    }
    void set(std::size_t pos) noexcept
    {
        const auto [ull_ind, bit_pos] = posInds(pos);
        m_data[ull_ind].value |= getMask(bit_pos);
    }
    void reset(std::size_t pos) noexcept
    {
        const auto [ull_ind, bit_pos] = posInds(pos);
        m_data[ull_ind].value &= ~getMask(bit_pos);
    }
    void assign(std::size_t pos, bool val) noexcept
    {
        const auto [ull_ind, bit_pos] = posInds(pos);
        m_data[ull_ind].value &= ~getMask(bit_pos);
        m_data[ull_ind].value |= static_cast< ull >(val) << bit_pos;
    }

    [[nodiscard]] bool atomicTest(std::size_t pos, std::memory_order order) noexcept
    {
        const auto [ull_ind, bit_pos] = posInds(pos);
        std::atomic_ref atomic_val{m_data[ull_ind].value};
        return atomic_val.load(order) & getMask(bit_pos);
    }
    void atomicSet(std::size_t pos, std::memory_order order) noexcept
    {
        const auto [ull_ind, bit_pos] = posInds(pos);
        std::atomic_ref atomic_val{m_data[ull_ind].value};
        atomic_val.fetch_or(getMask(bit_pos), order);
    }
    void atomicReset(std::size_t pos, std::memory_order order) noexcept
    {
        const auto [ull_ind, bit_pos] = posInds(pos);
        std::atomic_ref atomic_val{m_data[ull_ind].value};
        atomic_val.fetch_and(~getMask(bit_pos), order);
    }

    class BitReference
    {
        friend class DynamicBitset;

    public:
        BitReference& operator=(bool val)
        {
            target->assign(pos, val);
            return *this;
        }
        operator bool() { return target->test(pos); }

    private:
        BitReference(DynamicBitset* target_, std::size_t pos_) : target{target_}, pos{pos_} {}

        DynamicBitset* target;
        std::size_t    pos;
    };

    [[nodiscard]] bool         operator[](std::size_t pos) const noexcept { return test(pos); }
    [[nodiscard]] BitReference operator[](std::size_t pos) noexcept { return BitReference{this, pos}; }

private:
    struct AlignedUll
    {
        AlignedUll() = default;
        AlignedUll(ull v) : value{v} {}
        alignas(std::atomic_ref< ull >::required_alignment) ull value;
    };
    std::vector< AlignedUll > m_data;
    std::size_t               m_size;
};
} // namespace lstr
#endif // L3STER_UTIL_DYNAMICBITSET_HPP
