cmake_minimum_required(VERSION 3.10)
project(GLStudy)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

#find_package(GLUT REQUIRED)
#include_directories(${GLUT_INCLUDE_DIR})

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

include_directories(${GLFW_INCLUDE_DIR})

add_executable(GLStudy main.cpp)
target_link_libraries(GLStudy ${OPENGL_LIBRARY} ${GLFW_STATIC_LIBRARY})

