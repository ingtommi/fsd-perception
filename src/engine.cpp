/* 
Possible optimizations: 1. Use a separate stream for pre-processing and pipeline execution
                        2. Implement custom NMS to allow post-processing on GPU
                        3. Use TensorRT plugins to do pre- and post-processing without OpenCV
*/

#include "engine.hpp"
#include "timing.hpp"
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <yaml-cpp/yaml.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

static const cv::Scalar PAD_COLOR = {114, 114, 114};  // Padding color (constant gray)
static constexpr float NORM_FACTOR = 1.f / 255.f;     // Normalization factor

// Helper function to calculate tensor size from tensor shape
size_t getByteSizeByDim(const nvinfer1::Dims& dims) noexcept {
  size_t size = 1;
  for (int i = 0; i < dims.nbDims; i++) {
    size *= dims.d[i];
  }
  return size * sizeof(float);
}

// Constructor
YOLO::YOLO(const string& configPath) {
  // Parse config file
  if (!setOptions(configPath))
    throw runtime_error("Failed to parse config options from: " + configPath);
  // Open model file in binary mode
  ifstream modelFile(this->modelPath, ios::binary);
  if (!modelFile)
    throw runtime_error("Failed to open model from: " + modelPath); 
  // Determine file size and read into allocated memory
  modelFile.seekg(0, modelFile.end);  // go to end
  auto fsize = modelFile.tellg();     // get current position, which is the size of the file
  modelFile.seekg(0, modelFile.beg);  // go back to beginning
  vector<char> modelData(fsize);      // use vector for automatic memory management
  modelFile.read(modelData.data(), fsize);
  modelFile.close();

  // Create runtime, deserialize engine and initialize context using smart pointers
  runtime = unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!runtime)
    throw runtime_error("Failed to create inference runtime."); 
  engine = unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelData.data(), fsize));
  if (!engine)
    throw runtime_error("Failed to deserialize CUDA engine."); 
  context = unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
  if (!context) 
    throw runtime_error("Failed to create execution context."); 

  // Create CUDA stream
  cudaError_t status = cudaStreamCreate(&stream);
  if (status != cudaSuccess)
    throw runtime_error("Failed to create CUDA stream."); 

  this->buffers.resize(this->engine->getNbIOTensors()); // = 2
  // HACK: Use knowledge that YOLOs have 1 input ("images") and output ("output0") tensor to simplify code
  
  this->inputDims = engine->getTensorShape("images");
  this->pixels = this->inputDims.d[2] * this->inputDims.d[3]; // needed for pre-processing
  // Make sure input batch size is 1
  if (this->inputDims.d[0] != 1)
    throw runtime_error("Model requires batch size 1, but got batch size " + to_string(this->inputDims.d[0]));
  // Allocate CUDA memory for input tensor
  status = cudaMalloc(&this->buffers[0], getByteSizeByDim(this->inputDims));
  if (status != cudaSuccess)
    throw runtime_error("Failed to allocate CUDA memory for input.");  
  
  // Allocate CUDA memory for output tensor
  this->outputDims = engine->getTensorShape("output0");
  status = cudaMalloc(&this->buffers[1], getByteSizeByDim(this->outputDims));
  if (status != cudaSuccess) { 
    throw runtime_error("Failed to allocate CUDA memory for buffers."); 
  }
}

// Destructor (smart pointers automatically handle destruction)
YOLO::~YOLO() {
  // Destroy CUDA stream
  cudaStreamDestroy(stream);
  // Free each allocated buffer
  for (void* buffer : this->buffers) {
    if (buffer)
      cudaFree(buffer);
  }  
}

// Getter
vector<Object> YOLO::getDetections() noexcept {
  return this->detections;
}

// Config parser
bool YOLO::setOptions(const string& configPath) noexcept {
  try {
    // Load file
    YAML::Node config = YAML::LoadFile(configPath);
    
    //Extract options
    if (!config["model_path"]) return false;
    this->modelPath = config["model_path"].as<string>("");

    if (!config["model_type"]) return false;
    string modelTypeStr = config["model_type"].as<string>("");
    if (modelTypeStr == "YOLOv5")       
      this->modelType = ModelType::YOLOv5;
    else if (modelTypeStr == "YOLOv8" || modelTypeStr == "YOLOv11")  
      this->modelType = ModelType::YOLOv8;
    else if (modelTypeStr == "YOLOv10") 
      this->modelType = ModelType::YOLOv10;
    // Invalid model type
    else return false;

    if (!config["conf_thres"]) return false;
    this->confThres = config["conf_thres"].as<float>(0.f);
    if (this->confThres < 0.001f || this->confThres > 1.f) return false;

    if (!config["iou_thres"]) return false;
    this->iouThres = config["iou_thres"].as<float>(0.f);
    if (this->iouThres < 0.001f || this->iouThres > 1.f) return false;

    if (!config["max_det"]) return false;
    this->maxDet = config["max_det"].as<int>(0);
    if (this-> maxDet <= 0) return false;

    if (!config["nc"]) return false;
    this->numClasses = config["nc"].as<int>(0);
    if (this->numClasses <= 0) return false;
    
    if (!config["normalize"]) return false;
    this->zScore = config["normalize"].as<bool>(false);

    if (this->zScore) {
      if (!config["mean"] || !config["stddev"]) return false;
      vector<float> mean = config["mean"].as<vector<float>>(vector<float>({0.f, 0.f, 0.f}));
      this->zScoreMean = cv::Scalar(mean[0], mean[1], mean[2]); // BGR
      vector<float> std = config["stddev"].as<vector<float>>(vector<float>({1.f, 1.f, 1.f}));
      this->zScoreStd = cv::Scalar(std[0], std[1], std[2]); // BGR
    } 
  } catch (const exception& e) {
    return false;
  }
  return true;
}

// Pre-processing
cv::cuda::GpuMat YOLO::preProcess(const cv::Mat& frame) {
  // Define the OpenCV CUDA stream by wrapping the existing CUDA stream
  this->cvStream = cv::cuda::StreamAccessor::wrapStream(this->stream);
  
  // Upload frame to GPU
  cv::cuda::GpuMat gpuFrame;
  gpuFrame.upload(frame, cvStream);
  cv::cuda::GpuMat gpuModFrame = gpuFrame;

  // Resize and pad if needed
  if (frame.rows != this->inputDims.d[2] || frame.cols != this->inputDims.d[3]) {
    // Calculate scale factors
    float x_scale = static_cast<float>(frame.cols) / this->inputDims.d[3];
    float y_scale = static_cast<float>(frame.rows) / this->inputDims.d[2];
    this->scale = max(x_scale, y_scale);
    // Compute new dimensions
    int new_w = static_cast<int>(frame.cols / scale);
    int new_h = static_cast<int>(frame.rows / scale);
    // Resize frame
    cv::cuda::GpuMat gpuResizedFrame;
    cv::cuda::resize(gpuFrame, gpuResizedFrame, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR, this->cvStream);
    // Calculate padding required to match target size with frame at center 
    this->pad_top = (this->inputDims.d[2] - new_h) / 2;
    int pad_bot = this->inputDims.d[2] - new_h - this->pad_top;
    this->pad_left = (this->inputDims.d[3] - new_w) / 2;
    int pad_right = this->inputDims.d[3] - new_w - this->pad_left;
    // Add padding
    cv::cuda::copyMakeBorder(gpuResizedFrame, gpuModFrame, pad_top, pad_bot, pad_left, pad_right, 
                             cv::BORDER_CONSTANT, PAD_COLOR, this->cvStream);
  }

  // Normalize pixels to range [0,1]
  gpuModFrame.convertTo(gpuModFrame, CV_32FC3, NORM_FACTOR); // 1.f / 255.f
  
  if (this->zScore) {
    // Z-Score normalization
    cv::cuda::subtract(gpuModFrame, this->zScoreMean, gpuModFrame, cv::noArray(), -1, this->cvStream);
    cv::cuda::divide(gpuModFrame, this->zScoreStd, gpuModFrame, 1.0f, -1, this->cvStream);
  }
  
  // Convert BGR->RGB and NHWC->NCHW
  // 1: Allocate nchwFrame (output) as a flat, single-channel buffer that can store all three channels (pixels * 3)
  cv::cuda::GpuMat nchwFrame(1, this->pixels * 3, CV_32FC1);
  // 2: Create separate GpuMat objects for each channel that map to specific regions within nchwFrame
  vector<cv::cuda::GpuMat> channels{
    cv::cuda::GpuMat(this->inputDims.d[2], this->inputDims.d[3], CV_32FC1, (float*)nchwFrame.ptr() + 2 * pixels), // R channel
    cv::cuda::GpuMat(this->inputDims.d[2], this->inputDims.d[3], CV_32FC1, (float*)nchwFrame.ptr() + pixels),     // G channel
    cv::cuda::GpuMat(this->inputDims.d[2], this->inputDims.d[3], CV_32FC1, (float*)nchwFrame.ptr())               // B channel
  };
  // 3: Split gpuModFrame (input) into separate channels and copy them into the vector
  // When cv::cuda::split writes data into channels, it indirectly writes into the corresponding regions of nchwFrame
  cv::cuda::split(gpuModFrame, channels, this->cvStream);

  return nchwFrame;
}

// Post-processing
void YOLO::postProcess(cv::Mat& features) {
  // Select appropriate method
  switch (this->modelType) {
    // YOLOv5: "output0" = [1, 25200, (5 + numClasses)]
    case YOLOv5:
      postProcessV5(features);
      break;
    // YOLOv8 and YOLOv11: "output0" = [1, (4 + numClasses), 8400]
    case YOLOv8:
      postProcessV8(features);
      break;
    // YOLOv10: "output0" = [1, 300, 6]
    default:
      postProcessV10(features);
      break;
  }
}

void YOLO::postProcessV5(cv::Mat& features) {
  vector<cv::Rect> boxes;
  vector<float> scores;
  vector<int> labels;
        
  /* features = [x1, y1, w1, h1, conf1, class1_1, class2_1, ..., classNC_1,   
                 x2, y2, w2, h2, conf2, class1_2, class2_2, ..., classNC_2,   
                 .                                                            
                 .                                                            
                 .
                 xN, yN, wN, hN, confN, class1_N, class2_N, ..., classNC_N] (N = 25200) */
  
  // For each detection
  for (int i = 0; i < features.rows; ++i) {
    auto rowPtr = features.row(i).ptr<float>();
    // Get objectness score
    float objScore = *(rowPtr + 4);
    // Apply threshold
    if (objScore > this->confThres) {
      auto scoresPtr = rowPtr + 5;
      auto maxSPtr = max_element(scoresPtr, scoresPtr + this->numClasses);
      // Get class score and combine it with objectness score
      float clsScore = *maxSPtr;
      float score = objScore * clsScore;
      // Re-apply threshold
      if (score > this->confThres) {
        // Get label
        int label = maxSPtr - scoresPtr;
        // Get box coordinates (xywh format, normalized)
        float x = *rowPtr++; // get value and move pointer: x = *rowPtr; rowPtr++;
        float y = *rowPtr++; // alternative: x = *rowPtr, y = *(rowPtr+1), ...
        float w = *rowPtr++;
        float h = *rowPtr;
        // Convert coordinates
        int left = static_cast<int>(((x - 0.5 * w) - this->pad_left) * this->scale);
        int top = static_cast<int>(((y - 0.5 * h) - this->pad_top) * this->scale);
        int width = static_cast<int>(w * this->scale);
        int height = static_cast<int>(h * this->scale);
        // Store
        boxes.push_back(cv::Rect(left, top, width, height));
        scores.push_back(score);
        labels.push_back(label);
      }
    }
  }
  // Keep top maxDet detections (NMS)
  vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, this->confThres, this->iouThres, indices, 1.f, this->maxDet);
  // Collect final detections
  for (int id : indices) {
    this->detections.emplace_back(Object(boxes[id], labels[id], scores[id]));
  }
}

void YOLO::postProcessV8(cv::Mat& features) {
  vector<cv::Rect> boxes;
  vector<float> scores;
  vector<int> labels;

  /* features = [x1, x2, ..., xM, 
                 y1, y2, ..., yM, 
                 w1, w2, ..., wM, 
                 h1, h2, ..., hM, 
                 class1_1, class1_2, ..., class1_M, 
                 class2_1, class2_2, ..., class2_M,
                 .
                 .
                 .  
                 classNC_1, classNC_2, ..., classNC_M] (M = 8400)*/ 
  
  // Transpose matrix to have rows of features
  features = features.t();
  // For each detection
  for (int i = 0; i < features.rows; ++i) {
    auto rowPtr = features.row(i).ptr<float>();
    auto scoresPtr = rowPtr + 4;
    auto maxSPtr = max_element(scoresPtr, scoresPtr + this->numClasses);
    // Get confidence
    float score = *maxSPtr;
    // Apply threshold
    if (score > this->confThres) {
      // Get label
      int label = maxSPtr - scoresPtr;
      // Get box coordinates (xywh format, normalized)
      float x = *rowPtr++; // get value and move pointer
      float y = *rowPtr++;
      float w = *rowPtr++;
      float h = *rowPtr;
      // Convert coordinates
      int left = static_cast<int>(((x - 0.5 * w) - this->pad_left) * this->scale);
      int top = static_cast<int>(((y - 0.5 * h) - this->pad_top) * this->scale);
      int width = static_cast<int>(w * this->scale);
      int height = static_cast<int>(h * this->scale);
      // Store
      boxes.push_back(cv::Rect(left, top, width, height));
      scores.push_back(score);
      labels.push_back(label);
    }
  }
  // Keep top maxDet detections (NMS)
  vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, this->confThres, this->iouThres, indices, 1.f, this->maxDet);
  // Collect final detections
  for (int id : indices) {
    this->detections.emplace_back(Object(boxes[id], labels[id], scores[id]));
  }
}

void YOLO::postProcessV10(cv::Mat& features) {
  vector<cv::Rect> boxes;
  vector<float> scores;
  vector<int> labels;

  /* features = [x1_1, y1_1, x2_1, y2_1, conf1, id1, 
                 x1_2, y1_2, x2_2, y2_2, conf1, id1,
                 .
                 .
                 .
                 x1_N, y1_N, x2_N, y2_N, confN, idN] (N = 300) */ 
  
  // For each detection
  for (int i = 0; i < features.rows; ++i) {
    auto rowPtr = features.row(i).ptr<float>();
    // Get confidence
    float score = *(rowPtr + 4);
    // Apply threshold
    if (score > this->confThres) {
      // Get label
      int label = *(rowPtr + 5);
      // Get box coordinates (xyxy format, normalized)
      float x1 = *rowPtr++; // get value and move pointer
      float y1 = *rowPtr++;
      float x2 = *rowPtr++;
      float y2 = *rowPtr;
      // Convert coordinates
      int left = static_cast<int>((x1 - this->pad_left) * this->scale);
      int top = static_cast<int>((y1 - this->pad_top) * this->scale);
      int width = static_cast<int>((x2 - x1) * this->scale);
      int height = static_cast<int>((y2 - y1) * this->scale);
      // Store
      boxes.push_back(cv::Rect(left, top, width, height));
      scores.push_back(score);
      labels.push_back(label);
    }
  }
  // Keep top maxDet detections (no NMS)
  if (this->maxDet < static_cast<int>(labels.size())) {
    vector<int> indices(labels.size());
    // Populate indices
    iota(indices.begin(), indices.end(), 0);
    // Sort indices by confidence in descending order
    sort(indices.begin(), indices.end(), [&scores](int a, int b) {
      return scores[a] > scores[b];
    });
    // Keep only top maxDet indices
    indices.resize(this->maxDet);
    // Collect final detections
    for (int id : indices)
      this->detections.emplace_back(Object(boxes[id], labels[id], scores[id]));
  }
  else {
    for (size_t i = 0; i < labels.size(); ++i)
      this->detections.emplace_back(Object(boxes[i], labels[i], scores[i]));
  }
}

// Detection
// This function makes use of macros defined in 'timing.hpp' for benchmarking purpose. Inactive by default.
bool YOLO::infer(cv::Mat& frame) {

  // Declare timing variables
  INIT_TIMER

  // Pre-process input frame
  START_TIMER
  cv::cuda::GpuMat gpuFrame = preProcess(frame);
  END_TIMER("preprocess")

  START_TIMER
  // Copy input to buffer[0]
  cudaError_t status = cudaMemcpyAsync(this->buffers[0],           // input buffer  
                                       gpuFrame.data,              // pointer to data
                                       getByteSizeByDim(inputDims),
                                       cudaMemcpyDeviceToDevice,   // pre-processing on GPU
                                       this->stream);
  if (status != cudaSuccess) {
    spdlog::error("Error: Failed to copy data for inference.");
    return false;
  }

  // Set memory address for I/O
  this->context->setTensorAddress("images", this->buffers[0]);
  this->context->setTensorAddress("output0", this->buffers[1]);

  // Run inference
  this->context->enqueueV3(this->stream);

  // Copy output from buffer[1] into matrix of features
  cv::Mat features(this->outputDims.d[1], this->outputDims.d[2], CV_32F);
  status = cudaMemcpyAsync(features.data,                           
                           this->buffers[1],         // output buffer
                           getByteSizeByDim(outputDims),
                           cudaMemcpyDeviceToHost,   // post-processing on CPU
                           this->stream);
  if (status != cudaSuccess) { 
    spdlog::error("Error: Failed to copy data after inference.");
    return false;
  }

  // Synchronize stream to ensure all operations are complete
  cudaStreamSynchronize(this->stream);
  END_TIMER("inference")

  // Post-process output results
  START_TIMER
  this->detections.clear();
  postProcess(features);
  END_TIMER("postprocess")

  // Log timing results
  LOG_TIMER

  return true;
}

// Drawing
void YOLO::drawBbox(cv::Mat& frame, const vector<Object>& detections) {
  // For each detection
  for (const auto& detection : detections) {
    // Draw bounding box
    switch (detection.label) {
      case 0: // yellow cone
        cv::rectangle(frame, detection.bbox, cv::Scalar(0, 255, 255), 2);
        break;
      case 1: // blue cone
        cv::rectangle(frame, detection.bbox, cv::Scalar(255, 0, 0), 2);
        break;
      case 2: // Orange cone
        cv::rectangle(frame, detection.bbox,  cv::Scalar(0, 165, 255), 2);
        break;
      default: // Large orange (and unknown) cone
        cv::rectangle(frame, detection.bbox, cv::Scalar(0, 0, 255), 2);
        break;
    }
  }
}