add_executable( L3STER_benchmarks
                ${L3STER_DIR}/benchmarks/BenchmarksMain.cpp
                ${L3STER_DIR}/benchmarks/MeshBenchmarks.cpp
                ${L3STER_DIR}/benchmarks/ElementBenchmarks.cpp )

set( L3STER_BENCH_DATA_PATH "${L3STER_DIR}/benchmarks/data" CACHE STRING "Path to benchmark data directory" )
configure_file( ${L3STER_DIR}/benchmarks/DataPath.h.in DataPath.h )
target_include_directories( L3STER_benchmarks PRIVATE ${CMAKE_CURRENT_BINARY_DIR} )

find_package( benchmark REQUIRED )
target_link_libraries( L3STER_benchmarks PRIVATE L3STER benchmark::benchmark )