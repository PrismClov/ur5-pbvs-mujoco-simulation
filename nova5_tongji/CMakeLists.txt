cmake_minimum_required(VERSION 3.16)
project(nova5_tongji LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(UPSTREAM "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sp_vision_25")
if(NOT EXISTS "${UPSTREAM}/tools/extended_kalman_filter.cpp")
  message(FATAL_ERROR "Run git submodule update --init --recursive first")
endif()
if(EIGEN_INCLUDE_DIR)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIR}")
else()
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()
find_package(OpenCV REQUIRED COMPONENTS core)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
# Compile the actual upstream implementation, without source patches or a Python rewrite.
add_library(tongji_conveyor SHARED src/conveyor_adapter.cpp
  "${UPSTREAM}/tools/extended_kalman_filter.cpp"
  "${UPSTREAM}/tools/logger.cpp"
  "${UPSTREAM}/tools/math_tools.cpp"
  "${UPSTREAM}/tools/trajectory.cpp"
  "${UPSTREAM}/tasks/auto_aim/armor.cpp"
  "${UPSTREAM}/tasks/auto_aim/target.cpp"
  "${UPSTREAM}/tasks/auto_aim/aimer.cpp")
target_include_directories(tongji_conveyor PRIVATE
  "${UPSTREAM}"
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tongji_conveyor PRIVATE
  Eigen3::Eigen
  ${OpenCV_LIBS}
  fmt::fmt
  spdlog::spdlog
  yaml-cpp)
set_target_properties(tongji_conveyor PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
if(MINGW)
  target_link_options(tongji_conveyor PRIVATE -static -static-libgcc -static-libstdc++)
endif()
