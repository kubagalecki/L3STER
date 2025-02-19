#ifndef L3STER_UTIL_DYNAMICBITSET_HPP
#define L3STER_UTIL_DYNAMICBITSET_HPP

#include "l3ster/util/Common.hpp"

#include <atomic>
#include <bit>
#include <climits>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace lstr::util
{
class DynamicBitset
{
    static constexpr std::size_t ul_bits = sizeof(unsigned long) * CHAR_BIT;

    static std::ptrdiff_t getEntryInd(std::size_t pos) { return static_cast< std::ptrdiff_t >(pos / ul_bits); }
    static std::uint8_t   getBitPos(std::size_t pos) { return pos % ul_bits; }
    static unsigned long  getMask(std::uint8_t bit_pos) { return 0b1ul << bit_pos; }
    static auto           posInds(std::size_t pos) { return std::make_pair(getEntryInd(pos), getBitPos(pos)); }
    static std::size_t    computeAllocUls(std::size_t size)
    {
        return size % ul_bits == 0 ? size / ul_bits : size / ul_bits + 1;
    }
    [[nodiscard]] auto getUlAndMask(std::size_t pos) const
    {
        const auto [ul_ind, bit_pos] = posInds(pos);
        return std::make_pair(std::addressof(m_data[ul_ind]), getMask(bit_pos));
    }
    auto getUlAndMask(std::size_t pos)
    {
        const auto [ul_ind, bit_pos] = posInds(pos);
        return std::make_pair(std::addressof(m_data[ul_ind]), getMask(bit_pos));
    }

public:
    DynamicBitset() = default;
    DynamicBitset(std::size_t size) : m_data(computeAllocUls(size), 0ul), m_size{size} {}
    void resize(std::size_t new_size)
    {
        m_data.resize(computeAllocUls(new_size), 0ul);
        m_size = new_size;
    }

    [[nodiscard]] std::size_t count() const noexcept
    {
        return std::transform_reduce(
            begin(m_data), end(m_data), 0uz, std::plus<>{}, [](unsigned long v) { return std::popcount(v); });
    }
    [[nodiscard]] std::size_t size() const noexcept { return m_size; }

    [[nodiscard]] bool test(std::size_t pos) const noexcept
    {
        const auto [ul_ptr, mask] = getUlAndMask(pos);
        return *ul_ptr & mask;
    }
    void set(std::size_t pos) noexcept
    {
        const auto [ul_ptr, mask] = getUlAndMask(pos);
        *ul_ptr |= mask;
    }
    void reset(std::size_t pos) noexcept
    {
        const auto [ul_ptr, mask] = getUlAndMask(pos);
        *ul_ptr &= ~mask;
    }
    void flip(std::size_t pos) noexcept
    {
        const auto [ul_ptr, mask] = getUlAndMask(pos);
        *ul_ptr ^= mask;
    }
    void assign(std::size_t pos, bool val) noexcept
    {
        const auto [ul_ind, bit_pos] = posInds(pos);
        m_data[ul_ind] &= ~getMask(bit_pos);
        m_data[ul_ind] |= static_cast< unsigned long >(val) << bit_pos;
    }
    void clear() noexcept { std::ranges::fill(m_data, 0ul); }

    class BitReference
    {
        friend class DynamicBitset;

    public:
        BitReference& operator=(bool val)
        {
            m_target->assign(m_pos, val);
            return *this;
        }
        operator bool() const { return m_target->test(m_pos); }

    private:
        BitReference(DynamicBitset* target, std::size_t pos) : m_target{target}, m_pos{pos} {}

        DynamicBitset* m_target;
        std::size_t    m_pos;
    };

    [[nodiscard]] bool         operator[](std::size_t pos) const noexcept { return test(pos); }
    [[nodiscard]] BitReference operator[](std::size_t pos) noexcept { return BitReference{this, pos}; }

    class AtomicView
    {
        friend class DynamicBitset;
        AtomicView(DynamicBitset* target) : m_target{target} {}

    public:
        [[nodiscard]] bool test(std::size_t pos, std::memory_order order = std::memory_order_seq_cst) noexcept
        {
            const auto [ul_ptr, mask] = m_target->getUlAndMask(pos);
            return std::atomic_ref{*ul_ptr}.load(order) & mask;
        }
        void set(std::size_t pos, std::memory_order order = std::memory_order_seq_cst) noexcept
        {
            const auto [ul_ptr, mask] = m_target->getUlAndMask(pos);
            std::atomic_ref{*ul_ptr}.fetch_or(mask, order);
        }
        void reset(std::size_t pos, std::memory_order order = std::memory_order_seq_cst) noexcept
        {
            const auto [ul_ptr, mask] = m_target->getUlAndMask(pos);
            std::atomic_ref{*ul_ptr}.fetch_and(~mask, order);
        }
        void flip(std::size_t pos, std::memory_order order = std::memory_order_seq_cst) noexcept
        {
            const auto [ul_ptr, mask] = m_target->getUlAndMask(pos);
            std::atomic_ref{*ul_ptr}.fetch_xor(mask, order);
        }

    private:
        DynamicBitset* m_target;
    };
    [[nodiscard]] AtomicView getAtomicView() noexcept { return AtomicView{this}; }

    template < bool is_const >
    class SubView
    {
        using target_ptr_t = std::conditional_t< is_const, const DynamicBitset*, DynamicBitset* >;
        friend class DynamicBitset;
        SubView(target_ptr_t target, size_t begin, size_t end) : m_target{target}, m_begin{begin}, m_end{end} {}

    public:
        [[nodiscard]] bool test(std::size_t pos) const noexcept { return m_target->test(pos + m_begin); }
        void               set(std::size_t pos) noexcept
            requires(not is_const)
        {
            m_target->set(pos + m_begin);
        }
        void reset(std::size_t pos) noexcept
            requires(not is_const)
        {
            m_target->reset(pos + m_begin);
        }
        void flip(std::size_t pos) noexcept
            requires(not is_const)
        {
            m_target->flip(pos + m_begin);
        }
        void assign(std::size_t pos, bool val) noexcept
            requires(not is_const)
        {
            m_target->assign(pos + m_begin, val);
        }

        [[nodiscard]] size_t size() const noexcept { return m_end - m_begin; }

        BitReference operator[](size_t pos) noexcept
            requires(not is_const)
        {
            return m_target->operator[](pos + m_begin);
        }
        bool operator[](size_t pos) const noexcept
        {
            return const_cast< const DynamicBitset* >(m_target)->operator[](pos + m_begin);
        }

        [[nodiscard]] size_t count() const noexcept
        {
            if (m_begin == m_end)
                return 0;

            const auto [first_ul, first_bit] = posInds(m_begin);
            const auto [last_ul, last_bit]   = posInds(m_end - 1);

            if (first_ul == last_ul)
            {
                unsigned long mask{0};
                for (auto i = first_bit; i <= last_bit; ++i)
                    mask |= getMask(i);
                return std::popcount(m_target->m_data[first_ul] & mask);
            }
            else
            {
                size_t retval = 0;

                unsigned long mask{0};
                for (auto i = first_bit; i < ul_bits; ++i)
                    mask |= getMask(i);
                retval += std::popcount(m_target->m_data[first_ul] & mask);

                for (auto i = first_ul + 1; i < last_ul; ++i)
                    retval += std::popcount(m_target->m_data[i]);

                mask = 0;
                for (std::uint8_t i = 0; i <= last_bit; ++i)
                    mask |= getMask(i);
                retval += std::popcount(m_target->m_data[last_ul] & mask);

                return retval;
            }
        }

    private:
        target_ptr_t m_target;
        size_t       m_begin, m_end;
    };
    [[nodiscard]] SubView< false > getSubView(size_t begin, size_t end) noexcept
    {
        return SubView< false >{this, begin, end};
    }
    [[nodiscard]] SubView< true > getSubView(size_t begin, size_t end) const noexcept
    {
        return SubView< true >{this, begin, end};
    }

private:
    std::vector< unsigned long > m_data;
    std::size_t                  m_size{};
};

} // namespace lstr::util
#endif // L3STER_UTIL_DYNAMICBITSET_HPP
