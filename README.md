Here’s a complete and polished README for your **ResNet18 ONNX Inference in C++** project:

---

# **ResNet18 ONNX Inference in C++**

This project demonstrates how to load and run inference with a ResNet18 ONNX model in C++ using ONNX Runtime. It includes image preprocessing using OpenCV and outputs the top-5 predictions with their probabilities.

---

## **Prerequisites**

Ensure the following tools and libraries are installed:

- **CMake** (3.10 or higher)
- **C++ Compiler** with C++14 support
- **OpenCV** (for image loading and preprocessing)
- **ONNX Runtime** (for model inference)

---

## **Setup Instructions**

### **1. Install ONNX Runtime**

#### On Windows:
```bash
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

# Or with pip (Python bindings for testing)
pip install onnxruntime
```

#### On macOS:
```bash
# Using Homebrew
brew install onnxruntime

# Or with pip (Python bindings for testing)
pip install onnxruntime
```

---

### **2. Install OpenCV**

#### On Windows:
```bash
# Using vcpkg
vcpkg install opencv

# Or download prebuilt binaries from https://opencv.org/releases/
```

#### On Linux:
```bash
sudo apt-get install libopencv-dev
```

#### On macOS:
```bash
brew install opencv
```

---

### **3. Build the Project**

1. Clone this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/your-repo/resnet18-onnx-cpp.git
   cd resnet18-onnx-cpp
   ```

2. Create a build directory and configure the project using CMake:
   ```bash
   mkdir build && cd build
   cmake ..
   ```

3. If CMake cannot find ONNX Runtime or OpenCV, specify their paths manually:
   ```bash
   cmake -DONNXRUNTIME_INCLUDE_DIRS=/path/to/onnxruntime/include \
         -DONNXRUNTIME_LIBRARIES=/path/to/onnxruntime/lib/libonnxruntime.so \
         -DOpenCV_DIR=/path/to/opencv ..
   ```

4. Build the project:
   ```bash
   cmake --build .
   ```

---

## **Running the Inference**

### Basic Usage:
Run the compiled binary with the following arguments:

```bash
./resnet18_inference path/to/resnet18.onnx path/to/image.jpg path/to/imagenet_classes.txt
```

Example:
```bash
./resnet18_inference ../models/resnet18.onnx ../images/dog.jpg ../labels/imagenet_classes.txt
```

### Downloading ImageNet Labels:
If you don’t have a label file (`imagenet_classes.txt`), download it using this command:

```bash
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt -O imagenet_classes.txt
```

---

## **Example Output**

When you run the inference, you should see output similar to this:

```text
Loading model: resnet18.onnx
Loading image: dog.jpg
Running inference...
Top 5 Predictions:
1. golden retriever - 0.824352
2. Labrador retriever - 0.0715145
3. cocker spaniel - 0.0207977
4. kuvasz - 0.0144428
5. tennis ball - 0.0115891
```

---

## **Code Overview**

The program performs the following steps:

1. **Load Model:** The ResNet18 ONNX model is loaded using ONNX Runtime.
2. **Preprocess Image:** The input image is resized, normalized, and converted into a tensor format compatible with the model.
3. **Run Inference:** The preprocessed image tensor is passed through the model, and predictions are obtained.
4. **Postprocess Results:** The top-5 predictions are extracted along with their probabilities.

---

## **Troubleshooting**

If you encounter issues, check the following:

- **Memory Layout:** Ensure your ONNX model uses NCHW format (batch, channel, height, width). Most PyTorch-exported models use this format by default.
- **Input Dimensions:** Confirm that your input dimensions match what your model expects (default: `[1,4][224]`).
- **Preprocessing Steps:** Ensure that normalization values (mean/std) match those used during training.

---

## **Advanced Usage**

You can extend this project to include:

1. **Batch Processing:**
   Modify the input tensor dimensions to handle multiple images at once.
2. **Custom Input Sizes:**
   Adjust preprocessing to support models trained on non-standard resolutions.
3. **GPU Acceleration:**
   Use ONNX Runtime’s `CUDAExecutionProvider` for faster inference on NVIDIA GPUs.
4. **Quantized Models:**
   Use INT8 quantized models for faster inference on CPUs.

To enable GPU acceleration, modify the ONNX Runtime session initialization code in `main.cpp` as follows:

```cpp
Ort::SessionOptions session_options;
session_options.AppendExecutionProvider_CUDA(0); // Use CUDA GPU #0

Ort::Session session(env, "resnet18.onnx", session_options);
```

---

## **Performance Optimization Tips**

1. Use a release build (`cmake --build . --config Release`) for better performance.
2. Optimize preprocessing by parallelizing image transformations.
3. Consider using ONNX Runtime’s quantization tools for reduced latency.

---

## **Acknowledgments**

This project leverages the following open-source tools:

- [ONNX Runtime](https://github.com/microsoft/onnxruntime): High-performance runtime for ONNX models.
- [OpenCV](https://opencv.org/): Computer vision library used for image preprocessing.
- [PyTorch Hub](https://pytorch.org/hub/): Source of pre-trained ResNet18 models and ImageNet class labels.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Let me know if you’d like further customization or additional sections! 😊

---
Answer from Perplexity: pplx.ai/share