cmake_minimum_required(VERSION 2.8)

project(sandbox)

set(CMAKE_CXX_STANDARD 17)
include_directories(*.hpp *h)

add_executable(${PROJECT_NAME} "main.cpp")
