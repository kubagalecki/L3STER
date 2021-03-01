# Register L3STER tests with CTest
add_executable( L3STER_tests
                ${L3STER_DIR}/tests/TestsMain.cpp
                ${L3STER_DIR}/tests/MathTests.cpp
                ${L3STER_DIR}/tests/MeshTests.cpp
                ${L3STER_DIR}/tests/QuadratureTests.cpp
                )

set( L3STER_TESTS_DATA_PATH "${L3STER_DIR}/tests/data" CACHE STRING "Path to test data directory" )
configure_file( ${L3STER_DIR}/tests/TestDataPath.h.in TestDataPath.h )
target_include_directories( L3STER_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR} )

target_link_libraries( L3STER_tests
                       L3STER
                       Catch2::Catch2
                       )

catch_discover_tests( L3STER_tests )