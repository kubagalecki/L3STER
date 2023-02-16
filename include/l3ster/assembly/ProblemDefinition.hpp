#ifndef L3STER_ASSEMBLY_PROBLEMDEFINITION_HPP
#define L3STER_ASSEMBLY_PROBLEMDEFINITION_HPP

#include "l3ster/defs/Typedefs.h"
#include "l3ster/util/BitsetManip.hpp"
#include "l3ster/util/Common.hpp"

namespace lstr::detail
{
template < size_t n_fields, size_t n_domains >
using problem_def_t = std::array< Pair< d_id_t, std::array< bool, n_fields > >, n_domains >;

template < typename T >
inline constexpr bool is_problem_def = false;
template < size_t n_fields, size_t n_domains >
inline constexpr bool is_problem_def< problem_def_t< n_fields, n_domains > > = true;
template < typename T >
concept ProblemDef_c = is_problem_def< T >;

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

template < size_t n_fields, size_t n_domains >
constexpr size_t getFieldUllongSize(const problem_def_t< n_fields, n_domains >&)
{
    return bitsetNUllongs< n_fields >();
}

using empty_problem_def_t = std::array< Pair< d_id_t, std::array< bool, 0 > >, 0 >;
} // namespace lstr::detail
#endif // L3STER_ASSEMBLY_PROBLEMDEFINITION_HPP
