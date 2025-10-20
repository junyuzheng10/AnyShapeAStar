cmake_minimum_required(VERSION 3.16)
project(AnyShapeAStar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(anyshape_astar_demo
  src/main.cpp
)

target_link_libraries(anyshape_astar_demo
  ${OpenCV_LIBS}
)
