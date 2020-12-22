# Detect L3STER dependencies and configure accordingly
#
# This script aggregates calls to functions detecting the individual dependencies of L3STER.
#   Missing dependencies will be reported in a clear and user-friendly way, with tips on how
#   to obtain them.
#
include( "${L3STER_DIR}/scripts/cmake/DetectEigen.cmake" )
include( "${L3STER_DIR}/scripts/cmake/DetectTrilinos.cmake" )
include( "${L3STER_DIR}/scripts/cmake/DetectCatch2.cmake" )