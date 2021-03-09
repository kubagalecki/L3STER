# Detect L3STER dependencies and configure accordingly
#
# This script aggregates calls to functions detecting the individual dependencies of L3STER.
#   Missing dependencies will be reported in a clear and user-friendly way, with tips on how
#   to obtain them.
#
include( "${L3STER_DIR}/cmake/DetectEigen.cmake" )
include( "${L3STER_DIR}/cmake/DetectTrilinos.cmake" )
include( "${L3STER_DIR}/cmake/DetectCatch2.cmake" )
include( "${L3STER_DIR}/cmake/DetectHwloc.cmake" )
include( "${L3STER_DIR}/cmake/DetectTBB.cmake" )
