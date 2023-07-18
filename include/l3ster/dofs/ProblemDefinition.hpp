#ifndef L3STER_DOFS_PROBLEMDEFINITION_HPP
#define L3STER_DOFS_PROBLEMDEFINITION_HPP

#include "l3ster/defs/Typedefs.h"
#include "l3ster/util/BitsetManip.hpp"
#include "l3ster/util/Common.hpp"

namespace lstr
{
namespace detail
{
template < size_t n_fields, size_t n_domains >
using problem_def_t = std::array< util::Pair< d_id_t, std::array< bool, n_fields > >, n_domains >;

template < typename T >
inline constexpr bool is_problem_def = false;
template < size_t n_fields, size_t n_domains >
inline constexpr bool is_problem_def< problem_def_t< n_fields, n_domains > > = true;

template < size_t n_fields, size_t n_domains >
consteval size_t deduceNFields(const problem_def_t< n_fields, n_domains >&)
{
    return n_fields;
}

template < size_t n_fields, size_t n_domains >
consteval size_t deduceNDomains(const problem_def_t< n_fields, n_domains >&)
{
    return n_domains;
}
} // namespace detail

template < typename T >
concept ProblemDef_c = detail::is_problem_def< T >;

using EmptyProblemDef = std::array< util::Pair< d_id_t, std::array< bool, 0 > >, 0 >;
} // namespace lstr
#endif // L3STER_DOFS_PROBLEMDEFINITION_HPP
