cmake_minimum_required(VERSION 3.8)

project(sandbox)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(*.hpp)
include_directories(*.h)

add_executable(${PROJECT_NAME} "main.cpp")
