cmake_minimum_required(VERSION 3.13)
project(btlemon-distance)

add_subdirectory(btlemon)

include_directories(btlemon/include)
include_directories(cv-plot/CvPlot/inc)

add_compile_definitions(CVPLOT_HEADER_ONLY)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
find_package(Threads)

add_executable(bluetooth-distance src/main.cpp)
target_link_libraries(bluetooth-distance ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} btlemon)
