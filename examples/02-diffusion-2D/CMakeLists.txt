cmake_minimum_required( VERSION 3.24 )
project( diffusion-2D )
add_subdirectory( ../.. L3STER-bin )
add_executable( diffusion-2D source.cpp )
target_link_libraries( diffusion-2D L3STER )
