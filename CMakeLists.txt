
cmake_minimum_required(VERSION 3.10)
project(ResNet18ONNXInference)

# Use C++17 (required by ONNX Runtime)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Configure ONNX Runtime paths
set(ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-linux-x64-1.14.0")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

# Verify paths
if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
    message(FATAL_ERROR "ONNX Runtime library not found at: ${ONNXRUNTIME_LIBRARY}")
endif()

# 2. Find OpenCV properly with all necessary components
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn highgui imgcodecs)

# Print OpenCV information (helpful for debugging)
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# 3. Add executable
add_executable(resnet18_inference resnet18_onnx_inference.cpp)

# 4. Include directories with OpenCV4 header location
target_include_directories(resnet18_inference PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    # Add explicit opencv4 path for some distributions
    /usr/include/opencv4
)

# 5. Link libraries with modern CMake syntax
target_link_libraries(resnet18_inference PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_LIBRARY}"
)

# 6. Improved RPATH handling
set_target_properties(resnet18_inference PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)