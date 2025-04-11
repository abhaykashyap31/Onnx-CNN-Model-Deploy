# ResNet18 ONNX Inference in C++

This project demonstrates how to load and run inference with a ResNet18 ONNX model in C++ using ONNX Runtime.

## Prerequisites

- CMake (3.10 or higher)
- C++ compiler with C++14 support
- OpenCV (for image loading and preprocessing)
- ONNX Runtime (for model inference)

## Setup Instructions

### 1. Install ONNX Runtime

#### On Windows:

```
# Using vcpkg
vcpkg install onnxruntime

# Or manually:
# Download the Windows package from https://github.com/microsoft/onnxruntime/releases
# Extract it and set environment variables to point to the installation
```

#### On Linux:

```bash
# Ubuntu/Debian
sudo apt-get install libonnxruntime-dev

# Or with pip
pip install onnxruntime
```

#### On macOS:

```bash
# Using Homebrew
brew install onnxruntime

# Or with pip
pip install onnxruntime
```

### 2. Install OpenCV

#### On Windows:

```
# Using vcpkg
vcpkg install opencv

# Or download from https://opencv.org/releases/
```

#### On Linux:

```bash
sudo apt-get install libopencv-dev
```

#### On macOS:

```bash
brew install opencv
```

### 3. Build the Project

```bash
# Create a build directory
mkdir build && cd build

# Configure cmake (adjust paths as needed)
cmake ..

# If CMake cannot find ONNX Runtime, specify the path manually:
# cmake -DONNXRUNTIME_INCLUDE_DIRS=/path/to/onnxruntime/include -DONNXRUNTIME_LIBRARIES=/path/to/onnxruntime/lib/libonnxruntime.so ..

# Build
cmake --build .
```

## Running the Inference

```bash
# Basic usage
./resnet18_inference path/to/resnet18.onnx path/to/image.jpg path/to/imagenet_classes.txt

# If you don't have labels file, download it:
# wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

## Example Output

```
Loading model: resnet18.onnx
Loading image: example.jpg
Running inference...

Top 5 Predictions:
1. golden retriever - 0.824352
2. Labrador retriever - 0.0715145
3. cocker spaniel - 0.0207977
4. kuvasz - 0.0144428
5. tennis ball - 0.0115891
```

## Troubleshooting

- **Memory Layout**: Ensure your ONNX model was exported with the correct memory layout. This code assumes NCHW format (batch, channel, height, width).
- **Input Dimensions**: Make sure the input dimensions in the code match what your model expects. The default is [1, 3, 224, 224].
- **Preprocessing**: The preprocessing steps (normalization using ImageNet mean/std) should match how your model was trained.

## Advanced Usage

You can modify the code to support:

- Batch processing
- Different input sizes
- Different normalization values
- GPU acceleration by changing the execution provider
