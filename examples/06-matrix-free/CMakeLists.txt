cmake_minimum_required( VERSION 3.24 )
project( matrix-free )
add_subdirectory( ../.. L3STER-bin )
add_executable( matrix-free source.cpp )
target_link_libraries( matrix-free L3STER )
