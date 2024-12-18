cmake_minimum_required(VERSION 3.22)
project(PERCEPTION)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BENCHMARK "Enable benchmarking" OFF)
option(DETAIL "Enable detailed benchmarking" OFF)

# Find config files
find_file(CALIB "calibration.yaml" PATHS ${CMAKE_SOURCE_DIR}/config)
find_file(CAMERA "camera.yaml" PATHS ${CMAKE_SOURCE_DIR}/config)
find_file(YOLO "yolo.yaml" PATHS ${CMAKE_SOURCE_DIR}/config)
find_file(TASK "task.yaml" PATHS ${CMAKE_SOURCE_DIR}/config)
if(NOT CALIB)
  message(FATAL_ERROR "/config/calibration.yaml not found!")
endif()
if(NOT CAMERA)
  message(FATAL_ERROR "/config/camera.yaml not found!")
endif()
if(NOT YOLO)
  message(FATAL_ERROR "/config/yolo.yaml not found!")
endif()
if(NOT TASK)
  message(FATAL_ERROR "/config/task.yaml not found!")
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

# Find uEye
set(UEYE_INCLUDE_DIR /opt/ids/ueye/include)
set(UEYE_LIB_DIR /opt/ids/ueye/lib/aarch64-linux-gnu)

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${UEYE_INCLUDE_DIR}
)

# Add executable
if(BENCHMARK)
  # Use main() in benchmark.cpp
  add_executable(perception 
                src/benchmark.cpp src/engine.cpp src/depth.cpp src/camera.cpp)
  if(DETAIL)
    # Define BENCHMARK macro
    target_compile_definitions(perception PRIVATE DETAILED_BENCHMARK)
  endif()
else()
  # Use main() in main.cpp
  add_executable(perception 
                src/main.cpp src/engine.cpp src/depth.cpp src/camera.cpp)
endif()

# Link libraries
target_link_libraries(perception
  PUBLIC
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_INFER_PLUGIN}
    ${OpenCV_LIBRARIES}
    ${UEYE_LIB_DIR}/libueye_api.so
  PRIVATE
    yaml-cpp::yaml-cpp
    spdlog::spdlog
)

# Enable compiler warnings
target_compile_options(perception PRIVATE -Wall -Wextra -Werror)