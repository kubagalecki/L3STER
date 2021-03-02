# Create L3STER target
add_library( L3STER INTERFACE )
target_include_directories( L3STER INTERFACE ${L3STER_DIR}/include )

set_target_properties( L3STER PROPERTIES
                       INTERFACE_COMPILE_FEATURES cxx_std_20
                       INTERFACE_LINK_OPTIONS -pthread
                       )

target_link_libraries( L3STER INTERFACE
                       Eigen3::Eigen
                       Trilinos
                       hwloc
                       )

# Explicitly state L3STER sources
# This is helpful to the IDE, but not required for a correct build
file( GLOB_RECURSE L3STER_SOURCES
      LIST_DIRECTORIES false
      RELATIVE ${L3STER_DIR}
      CONFIGURE_DEPENDS
      include/*
      )
target_sources( L3STER INTERFACE ${L3STER_SOURCES} )