#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// Function to load labels from a file
std::vector<std::string> loadLabels(const std::string& filename) {
    std::vector<std::string> labels;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open labels file: " << filename << std::endl;
        return labels;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

// Preprocess image for ResNet18
std::vector<float> preProcessImage(cv::Mat& image) {
    // Resize and crop
    cv::resize(image, image, cv::Size(256, 256));
    cv::Rect roi(16, 16, 224, 224); // Center crop
    image = image(roi);
    
    // Convert to float and normalize
    std::vector<float> input(3 * 224 * 224);
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[] = {0.229f, 0.224f, 0.225f};
    
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 224; h++) {
            for (int w = 0; w < 224; w++) {
                float pixel = static_cast<float>(image.at<cv::Vec3b>(h, w)[c]) / 255.0f;
                input[c * 224 * 224 + h * 224 + w] = (pixel - mean[c]) / std[c];
            }
        }
    }
    return input;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_model> <path_to_image> [path_to_labels]" << std::endl;
        return 1;
    }
    
    const std::string modelPath = argv[1];
    const std::string imagePath = argv[2];
    const std::string labelsPath = argc > 3 ? argv[3] : "imagenet_classes.txt";
    
    try {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ResNet18Inference");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load model
        std::cout << "Loading model: " << modelPath << std::endl;
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
        
        // Get model input/output information
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Fixed: Use GetInputNameAllocated/GetOutputNameAllocated for ONNX Runtime >=1.13
        auto inputNameAllocated = session.GetInputNameAllocated(0, allocator);
        const char* inputName = inputNameAllocated.get();
        
        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
        
        auto outputNameAllocated = session.GetOutputNameAllocated(0, allocator);
        const char* outputName = outputNameAllocated.get();
        
        // Load and preprocess image
        std::cout << "Loading image: " << imagePath << std::endl;
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            return 1;
        }
        
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        std::vector<float> inputTensorValues = preProcessImage(image);
        
        // Prepare input/output names
        const std::vector<const char*> inputNames{inputName};
        const std::vector<const char*> outputNames{outputName};
        
        // Create input tensor
        const std::vector<int64_t> inputShape = {1, 3, 224, 224};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, 
            inputTensorValues.data(), 
            inputTensorValues.size(),
            inputShape.data(), 
            inputShape.size()
        );
        
        // Run inference
        std::cout << "Running inference..." << std::endl;
        auto outputTensors = session.Run(
            Ort::RunOptions{nullptr}, 
            inputNames.data(), 
            &inputTensor, 
            1, 
            outputNames.data(), 
            1
        );
        
        // Process results
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        const int64_t numClasses = outputShape[1];
        
        // Softmax calculation
        std::vector<float> probabilities(numClasses);
        float sum = 0.0f;
        for (int64_t i = 0; i < numClasses; i++) {
            probabilities[i] = std::exp(outputData[i]);
            sum += probabilities[i];
        }
        std::transform(probabilities.begin(), probabilities.end(), probabilities.begin(),
                      [sum](float val) { return val / sum; });
        
        // Get top 5 predictions
        std::vector<std::pair<float, int64_t>> scores;
        for (int64_t i = 0; i < numClasses; i++) {
            scores.emplace_back(probabilities[i], i);
        }
        
        std::partial_sort(scores.begin(), scores.begin() + 5, scores.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Load and print labels
        const std::vector<std::string> labels = loadLabels(labelsPath);
        std::cout << "\nTop 5 Predictions:" << std::endl;
        for (int i = 0; i < 5; i++) {
            const int64_t classId = scores[i].second;
            const float probability = scores[i].first;
            const std::string className = (classId < labels.size()) 
                                        ? labels[classId] 
                                        : "Unknown";
            std::cout << i + 1 << ". " << className 
                     << " - " << probability << std::endl;
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
