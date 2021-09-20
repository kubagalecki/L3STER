add_executable( L3STER_benchmarks BenchmarksMain.cpp
                MeshBenchmarks.cpp
                ElementBenchmarks.cpp
                )

target_include_directories( L3STER_benchmarks PRIVATE ${CMAKE_CURRENT_BINARY_DIR} )

find_package( benchmark REQUIRED )
target_link_libraries( L3STER_benchmarks PRIVATE L3STER benchmark::benchmark )

set( L3STER_BENCH_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data" )
configure_file( DataPath.h.in DataPath.h )