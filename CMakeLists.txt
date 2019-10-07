cmake_minimum_required(VERSION 3.10)
project(canny)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PREFIX_PATH "/opencv3.4.1/usr/")
find_package(OpenCV 3.4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(canny main.cpp)
add_executable(match match.cpp)
add_executable(3d2d 3d2d.cpp)
add_executable(LK LK.cpp)

target_link_libraries(canny ${OpenCV_LIBS})
target_link_libraries(match ${OpenCV_LIBS})
target_link_libraries(3d2d ${OpenCV_LIBS})
target_link_libraries(LK ${OpenCV_LIBS})