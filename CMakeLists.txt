cmake_minimum_required(VERSION 3.22)
project(YOLO-TRT)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BENCHMARK "Enable benchmarking" OFF)
option(DETAIL "Enable detailed benchmarking" OFF)

# Find config file
find_file(CONFIG "config.yaml" PATHS ${CMAKE_SOURCE_DIR})
if(NOT CONFIG)
  message(FATAL_ERROR "config.yaml not found!")
endif()

# Find CUDA, OpenCV, yaml-cpp and spdlog
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)

# Find TensorRT
set(TENSORRT_INCLUDE_DIR /usr/include/aarch64-linux-gnu)
set(TENSORRT_LIB_DIR /usr/lib/aarch64-linux-gnu)
# Find TensorRT libraries
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_LIB_DIR}
  REQUIRED)

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

# Add executable
if(BENCHMARK)
  # Use main() in benchmark.cpp
  add_executable(yolo_detector 
                src/benchmark.cpp src/engine.cpp)
  if(DETAIL)
    # Define BENCHMARK macro
    target_compile_definitions(yolo_detector PRIVATE DETAILED_BENCHMARK)
  endif()
else()
  # Use main() in main.cpp
  add_executable(yolo_detector 
                src/main.cpp src/engine.cpp)
endif()

# Link libraries
target_link_libraries(yolo_detector
  PUBLIC
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_INFER_PLUGIN}
    ${OpenCV_LIBRARIES}
  PRIVATE
    yaml-cpp::yaml-cpp
    spdlog::spdlog
)

# Enable compiler warnings
target_compile_options(yolo_detector PRIVATE -Wall -Wextra -Werror)