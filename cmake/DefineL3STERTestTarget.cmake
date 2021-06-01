add_executable( L3STER_tests
                ${L3STER_DIR}/tests/TestsMain.cpp
                ${L3STER_DIR}/tests/MathTests.cpp
                ${L3STER_DIR}/tests/MeshTests.cpp
                ${L3STER_DIR}/tests/QuadratureTests.cpp
                ${L3STER_DIR}/tests/HwlocTests.cpp
                ${L3STER_DIR}/tests/NodeAllocationTests.cpp
                ${L3STER_DIR}/tests/MappingTests.cpp )

set( L3STER_TESTS_DATA_PATH "${L3STER_DIR}/tests/data" )
execute_process( COMMAND bash -c "chmod +x ${L3STER_DIR}/scripts/n_numa_nodes.sh" )
execute_process( COMMAND bash ${L3STER_DIR}/scripts/n_numa_nodes.sh
                 OUTPUT_VARIABLE L3STER_N_NUMA_NODES )
configure_file( ${L3STER_DIR}/tests/TestDataPath.h.in TestDataPath.h )
target_include_directories( L3STER_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR} )
target_link_libraries( L3STER_tests
                       L3STER
                       Catch2::Catch2
                       )

catch_discover_tests( L3STER_tests )

include( ${L3STER_DIR}/cmake/RegisterMpiTests.cmake )
include( ${L3STER_DIR}/cmake/GenerateCoverageScript.cmake )