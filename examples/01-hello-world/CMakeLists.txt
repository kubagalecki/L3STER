cmake_minimum_required( VERSION 3.24 )
project( hello-world-l3ster )
add_subdirectory( ../.. L3STER-bin )
add_executable( hello-world source.cpp )
target_link_libraries( hello-world L3STER )