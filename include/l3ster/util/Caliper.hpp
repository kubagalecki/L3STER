#ifndef L3STER_UTIL_CALIPER_HPP
#define L3STER_UTIL_CALIPER_HPP

#ifdef L3STER_PROFILE_EXECUTION
#include "caliper/cali.h"
#define L3STER_PROFILE_REGION_BEGIN(name) CALI_MARK_BEGIN(name)
#define L3STER_PROFILE_REGION_END(name) CALI_MARK_END(name)
#define L3STER_PROFILE_FUNCTION CALI_CXX_MARK_FUNCTION
#else
#define L3STER_PROFILE_REGION_BEGIN(name)
#define L3STER_PROFILE_REGION_END(name)
#define L3STER_PROFILE_FUNCTION
#endif

#endif // L3STER_UTIL_CALIPER_HPP