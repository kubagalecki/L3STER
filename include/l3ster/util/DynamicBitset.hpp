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

namespace lstr
{
class DynamicBitset
{
    static constexpr std::size_t ul_bits = sizeof(unsigned long) * CHAR_BIT;

    static inline std::ptrdiff_t getEntryInd(std::size_t pos) { return static_cast< std::ptrdiff_t >(pos / ul_bits); }
    static inline std::uint8_t   getBitPos(std::size_t pos) { return pos % ul_bits; }
    static inline unsigned long  getMask(std::uint8_t bit_pos) { return 0b1ul << bit_pos; }
    static inline auto           posInds(std::size_t pos) { return std::make_pair(getEntryInd(pos), getBitPos(pos)); }
    static inline std::size_t    computeAllocUls(std::size_t size)
    {
        return size % ul_bits == 0 ? size / ul_bits : size / ul_bits + 1;
    }

public:
    DynamicBitset() = default;
    DynamicBitset(std::size_t size_) : m_data(computeAllocUls(size_), 0ul), m_size{size_} {}
    void resize(std::size_t new_size)
    {
        m_data.resize(computeAllocUls(new_size), 0ul);
        m_size = new_size;
    }

    [[nodiscard]] std::size_t count() const noexcept
    {
        return std::transform_reduce(
            begin(m_data), end(m_data), std::size_t{0}, std::plus<>{}, [](CacheAligned< unsigned long > v) {
                return std::popcount(*v);
            });
    }
    [[nodiscard]] std::size_t size() const noexcept { return m_size; }

    [[nodiscard]] bool test(std::size_t pos) const noexcept
    {
        const auto [ul_ind, bit_pos] = posInds(pos);
        return *m_data[ul_ind] & getMask(bit_pos);
    }
    void set(std::size_t pos) noexcept
    {
        const auto [ul_ind, bit_pos] = posInds(pos);
        *m_data[ul_ind] |= getMask(bit_pos);
    }
    void reset(std::size_t pos) noexcept
    {
        const auto [ul_ind, bit_pos] = posInds(pos);
        *m_data[ul_ind] &= ~getMask(bit_pos);
    }
    void flip(std::size_t pos) noexcept
    {
        const auto [ul_ind, bit_pos] = posInds(pos);
        *m_data[ul_ind] ^= getMask(bit_pos);
    }
    void assign(std::size_t pos, bool val) noexcept
    {
        const auto [ul_ind, bit_pos] = posInds(pos);
        *m_data[ul_ind] &= ~getMask(bit_pos);
        *m_data[ul_ind] |= static_cast< unsigned long >(val) << bit_pos;
    }
    void clear() noexcept { std::ranges::fill(m_data, CacheAligned< unsigned long >{0ul}); }

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
            const auto [ul_ind, bit_pos] = posInds(pos);
            std::atomic_ref atomic_val{*m_target->m_data[ul_ind]};
            return atomic_val.load(order) & getMask(bit_pos);
        }
        void set(std::size_t pos, std::memory_order order = std::memory_order_seq_cst) noexcept
        {
            const auto [ul_ind, bit_pos] = posInds(pos);
            std::atomic_ref atomic_val{*m_target->m_data[ul_ind]};
            atomic_val.fetch_or(getMask(bit_pos), order);
        }
        void reset(std::size_t pos, std::memory_order order = std::memory_order_seq_cst) noexcept
        {
            const auto [ul_ind, bit_pos] = posInds(pos);
            std::atomic_ref atomic_val{*m_target->m_data[ul_ind]};
            atomic_val.fetch_and(~getMask(bit_pos), order);
        }
        void flip(std::size_t pos, std::memory_order order = std::memory_order_seq_cst) noexcept
        {
            const auto [ul_ind, bit_pos] = posInds(pos);
            std::atomic_ref atomic_val{*m_target->m_data[ul_ind]};
            atomic_val.fetch_xor(getMask(bit_pos), order);
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
                return std::popcount(*m_target->m_data[first_ul] & mask);
            }
            else
            {
                size_t retval = 0;

                unsigned long mask{0};
                for (auto i = first_bit; i < ul_bits; ++i)
                    mask |= getMask(i);
                retval += std::popcount(*m_target->m_data[first_ul] & mask);

                for (auto i = first_ul + 1; i < last_ul; ++i)
                    retval += std::popcount(*m_target->m_data[i]);

                mask = 0;
                for (std::uint8_t i = 0; i <= last_bit; ++i)
                    mask |= getMask(i);
                retval += std::popcount(*m_target->m_data[last_ul] & mask);

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
    std::vector< CacheAligned< unsigned long > > m_data;
    std::size_t                                  m_size{};
};
} // namespace lstr
#endif // L3STER_UTIL_DYNAMICBITSET_HPP
