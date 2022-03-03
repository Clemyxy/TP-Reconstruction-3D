cmake_minimum_required(VERSION 3.20)
project(3D_Reconstruct)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

set(SRCS src/main.cpp)

add_executable(3D_Reconstruct ${SRCS})
target_link_libraries(3D_Reconstruct ${OpenCV_LIBS})

add_executable(NoCalibReconstruct src/nocalib.cpp)
target_link_libraries(NoCalibReconstruct ${OpenCV_LIBS})