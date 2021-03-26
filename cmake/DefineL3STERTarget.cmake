add_library( L3STER INTERFACE )
target_include_directories( L3STER INTERFACE ${L3STER_DIR}/include )
set_target_properties( L3STER PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_20 )
target_link_libraries( L3STER INTERFACE ${L3STER_DEPENDENCY_LIST} )

# Explicitly state L3STER sources - this is helpful to the IDE, but not required for a correct build
file( GLOB_RECURSE L3STER_SOURCES
      LIST_DIRECTORIES false
      RELATIVE ${L3STER_DIR}
      CONFIGURE_DEPENDS
      include/*
      )
target_sources( L3STER INTERFACE ${L3STER_SOURCES} )