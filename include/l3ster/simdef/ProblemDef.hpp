#ifndef L3STER_SIMSTRUCTURE_PROBLEMDEF_HPP
#define L3STER_SIMSTRUCTURE_PROBLEMDEF_HPP

#include <utility>

#include "IOGraphDef.hpp"
#include "SubiterDef.hpp"

namespace lstr::def
{
template < Subiter_c... S >
struct Problem
{
    constexpr Problem(const SubiterSet< S... >& subiters_, IOGraph io_ = {})
        : subiters{subiters_.subiters}, io{std::move(io_)}
    {}

    std::tuple< S... > subiters;
    IOGraph            io{};
};

template < Subiter_c S >
Problem(S) -> Problem< S >;

namespace detail
{
template < typename T >
constexpr bool is_problem = false;

template < Subiter_c... S >
constexpr bool is_problem< Problem< S... > > = true;
} // namespace detail

template < typename T >
concept Problem_c = detail::is_problem< T >;

template < typename... P >
struct ProblemSet
{
    constexpr ProblemSet(const P&... problems_) : problems{problems_...} {}

    std::tuple< P... > problems;
};
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_PROBLEMDEF_HPP
