find_package( benchmark REQUIRED )
include( CTest )

add_executable( Non_MPI_benchmarks BenchmarksMain.cpp
                ElementBenchmarks.cpp
                GlobalAssemblyBenchmarks.cpp
                LocalAssemblyBenchmarks.cpp
                MeshBenchmarks.cpp
                )
target_include_directories( Non_MPI_benchmarks PRIVATE ${CMAKE_CURRENT_BINARY_DIR} )
target_link_libraries( Non_MPI_benchmarks PRIVATE L3STER benchmark::benchmark )

file( MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/non_MPI_benchmarks_wd" )
add_test( NAME "Non_MPI_benchmarks"
          COMMAND $<TARGET_FILE:Non_MPI_benchmarks> --benchmark_counters_tabular=true
          WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/non_MPI_benchmarks_wd"
          )
set_tests_properties( Non_MPI_benchmarks PROPERTIES ENVIRONMENT "OMP_PROC_BIND=false" )

set( L3STER_BENCH_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data" )
configure_file( DataPath.h.in DataPath.h )

find_package( caliper REQUIRED )
function( addMpiBenchmark source name nprocs )
    get_filename_component( bench_target ${source} NAME_WE )
    add_executable( ${bench_target} ${source} )
    target_include_directories( ${bench_target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} )
    target_link_libraries( ${bench_target} L3STER caliper )
    target_compile_options( ${bench_target} PRIVATE "-DL3STER_PROFILE_EXECUTION" )
    set( bench_env "OMP_PROC_BIND=false;CALI_CONFIG=runtime-report,mem.highwatermark,calc.inclusive" )
    set( bench_driver mpiexec )
    if ( ${ARGC} EQUAL 4 )
        set( args ${ARGV4} )
    endif ()
    foreach ( np ${nprocs} )
        set( bench_args "--nooversubscribe;--bind-to;socket;--map-by;ppr:1:socket;-np;${np};$<TARGET_FILE:${bench_target}>;${args}" )
        set( bench_name ${name}_np_${np} )
        set( bench_dir "${CMAKE_CURRENT_BINARY_DIR}/${bench_name}" )
        file( MAKE_DIRECTORY ${bench_dir} )
        list( APPEND L3STER_MPI_BENCHMARKS ${bench_name} )
        add_test( NAME ${bench_name}
                  COMMAND ${bench_driver} ${bench_args}
                  WORKING_DIRECTORY ${bench_dir}
                  )
        set_tests_properties( ${bench_name} PROPERTIES ENVIRONMENT "${bench_env}" )
    endforeach ()
    set( L3STER_MPI_BENCHMARKS ${L3STER_MPI_BENCHMARKS} PARENT_SCOPE )
endfunction()

addmpibenchmark( Diffusion3DBenchmark.cpp Solve_diffusion_problem_3D "1;2;4;8" )