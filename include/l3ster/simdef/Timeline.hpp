#ifndef L3STER_TIMELINE_HPP
#define L3STER_TIMELINE_HPP

#include "l3ster/defs/Typedefs.h"
#include "l3ster/util/ConstexprRefStableCollection.hpp"

#include <string_view>
#include <tuple>

namespace lstr::def
{
template < typename... Kernels >
class Timeline;

template < typename T >
struct Kernel
{
    std::string_view name;
    T                obj;
};
template < typename T >
Kernel(std::string_view, T&&) -> Kernel< std::remove_cvref_t< T > >;

template < typename... T >
class KernelSet
{
    friend class Timeline< T... >;
    struct ObjectAccessToken
    {
        size_t index;
    };

public:
    constexpr KernelSet(const Kernel< T >&... objects) : m_objects{objects.obj...}, m_names{objects.name...}
    {
        auto name_copy = m_names;
        std::ranges::sort(name_copy);
        if (std::ranges::adjacent_find(name_copy) != end(name_copy))
            throw std::logic_error{"Object names must be unique"};
    }
    template < ObjectAccessToken token >
    [[nodiscard]] constexpr const auto& getObject() const
    {
        return std::get< token.index >(m_objects);
    }
    [[nodiscard]] constexpr auto getObjectAccessToken(std::string_view name) const
    {
        const auto it = std::ranges::find(m_names, name);
        if (it == end(m_names))
            throw std::logic_error{"No object has this name"};
        return ObjectAccessToken{static_cast< size_t >(std::distance(begin(m_names), it))};
    }

private:
    std::tuple< T... >                           m_objects;
    std::array< std::string_view, sizeof...(T) > m_names;
};
template < typename... T >
KernelSet(const Kernel< T >&...) -> KernelSet< T... >;

template < typename... Kernels >
class Timeline
{
    using kernel_tuple_t = KernelSet< Kernels... >;
    using kernel_token_t = typename kernel_tuple_t::ObjectAccessToken;
    struct Field
    {
        std::string_view          name;
        std::uint8_t              n_components{};
        ConstexprVector< d_id_t > domains;
    };
    struct Equation
    {
        std::string_view                    name;
        kernel_token_t                      kernel_token;
        ConstexprVector< std::string_view > fields;
        ConstexprVector< d_id_t >           domains;
        q_o_t                               max_val_order{}, max_der_order{};
    };
    struct BoundaryCondition
    {
        std::string_view                    name;
        kernel_token_t                      kernel_token;
        ConstexprVector< std::string_view > fields;
        ConstexprVector< d_id_t >           domains;
        q_o_t                               max_val_order{}, max_der_order{};
    };
    struct System
    {
        ConstexprVector< const Equation* >          domain_physics;
        ConstexprVector< const BoundaryCondition* > boundary_physics;
    };

public:
    constexpr Timeline() = default;
    constexpr Timeline(const kernel_tuple_t& kernels) : m_kernels{kernels} {}

    constexpr const Field* addField(std::string_view name, std::uint8_t n_components, ConstexprVector< d_id_t > domains)
    {
        return std::addressof(m_fields.emplace(name, n_components, std::move(domains)));
    }

private:
    kernel_tuple_t m_kernels;

    ConstexprRefStableCollection< Field >             m_fields;
    ConstexprRefStableCollection< Equation >          m_dom_kernels;
    ConstexprRefStableCollection< BoundaryCondition > m_boundary_kernels;
};
template < typename... Kernels >
Timeline(KernelSet< Kernels... >) -> Timeline< Kernels... >;
Timeline() -> Timeline<>;
} // namespace lstr::def
#endif // L3STER_TIMELINE_HPP
