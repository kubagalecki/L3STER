# ---  detect_trilinos  ---
# Detect if Trilinos is installed and was built with the required packages
#
include( ${L3STER_DIR}/cmake/DefineTrilinosTarget.cmake )
define_trilinos_target( ${L3STER_ENABLE_VERBOSITY} Tpetra Belos MueLu )