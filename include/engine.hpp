#ifndef ENGINE_H
#define ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <spdlog/spdlog.h>
#include <NvInfer.h>
//#include <NvInferSafeRuntime.h> // Use this for strict safety requirements (i.e. vehicles) [namespace -> nvinfer1::safe]
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

/**
 * @brief Logger class for TensorRT.
 * 
 * This class inherits from nvinfer1::ILogger and provides a logging mechanism for TensorRT.
 */
class Logger : public nvinfer1::ILogger {
 public:
    /**
     * @brief Logs a message with a given severity.
     * 
     * @param severity The severity level of the message.
     * @param msg The message to log.
     */
  void log(Severity severity, const char* msg) noexcept override {
    if (severity == Severity::kWARNING)   // WARNING
      spdlog::warn("[TRT] {}", msg);
    else if (severity <= Severity::kERROR) // ERROR or INTERNAL_ERROR
      spdlog::error("[TRT] {}", msg);
  }
};

/**
 * @brief Object detection class.
 * 
 * This class performs object detection using YOLO.
 */
class YOLO {
  public:
    /**
     * @brief Struct to hold detection results.
     * 
     * This struct holds the bounding box, confidence score, and label of a detection result.
     */
    struct Object {
      const cv::Rect bbox; // Bounding box of the detected object.
      const int label;     // Label of the detected object.
      const float conf;    // Confidence score of the detection.

      /**
       * @brief Constructs a DetResult object.
       * 
       * @param bbox Bounding box of the detected object.
       * @param label Label of the detected object.
       * @param conf Confidence score of the detection.
       */
      Object(const cv::Rect& bbox, const int& label, const float& conf) : bbox(bbox), label(label), conf(conf) {}
    };

    /**
     * @brief Constructs a YOLO object from config file path.
     * 
     * @param configPath Path to the config file specifying the model to use and inference options.
     */
    YOLO(const std::string& configPath);

    /** 
     * @brief Destructs a YOLO object.
     */
    ~YOLO();
    
    /**
     * @brief Getter for the vector of detected objects.
     * 
     * @return Reference to the vector of detected objects.
     */
    const std::vector<Object>& getDetections() noexcept;

    /**
     * @brief Performs inference on a frame.
     * 
     * @param frame Frame on which to perform inference.
     * @return True if successful, False otherwise.
     */
    bool infer(cv::Mat& frame);

  private:
    // YOLO options
    std::string modelPath;                               // Path to YOLO model.
    enum ModelType { 
      YOLOv5, 
      YOLOv8, // also includes YOLOv11 
      YOLOv10
    } modelType;                                          // Type of model.
    float confThres;                                      // Confidence threshold for detection.
    float iouThres;                                       // IoU threshold for NMS.                       
    int maxDet;                                           // Max detections.
    bool zScore;                                          // Flag to enable Z-Score normalization
    cv::Scalar zScoreMean;                                // Mean values for Z-Score normalization.
    cv::Scalar zScoreStd;                                 // StdDev values for Z-Score normalization.
    int numClasses;                                       // Number of detection classes.

    // Pre- and post-processing
    float scale = 1.f;                                    // Scale factor for resizing, default = 1.
    size_t pixels;                                        // Number of pixels per channel (h * w)
    int pad_top = 0;                                      // Padding value for top of frame, default = 0;
    int pad_left = 0;                                     // Padding value for left of frame, default = 0;

    // Results
    std::vector<Object> detections;                       // Vector of detected objects.
    
    // CUDA stream for asynchronous operations
    cudaStream_t stream;                                  // CUDA stream for async operations.
    cv::cuda::Stream cvStream;                            // OpenCV CUDA stream representing the default stream.

    // TensorRT
    Logger logger;                                        // Logger instance for TensorRT.
    std::unique_ptr<nvinfer1::IRuntime> runtime;          // Runtime for the model.
    std::unique_ptr<nvinfer1::ICudaEngine> engine;        // Engine for inference.
    std::unique_ptr<nvinfer1::IExecutionContext> context; // Execution context for inference.
    nvinfer1::Dims inputDims;                             // Dims of input tensor      
    nvinfer1::Dims outputDims;                            // Dims of output tensor
    std::vector<void*> buffers;                           // Pointers to GPU memory locations that hold input and output tensors.
    
    /**
     * @brief Loads inference options from config path.
     * 
     * @param configPath Path to the config file.
     * @return True if successful, False otherwise.
     */
    bool setOptions(const std::string& configPath) noexcept;

    /**
     * @brief Preprocesses an input frame by performing letterboxing and normalization.
     * 
     * @param frame Frame to be pre-processed.
     * @returns A preprocessed version of the input frame as a CUDA GPU matrix.
     */
    cv::cuda::GpuMat preProcess(const cv::Mat& frame);

    /**
     * @brief Postprocesses the detection results.
     * 
     * @param features Feature matrix containing raw detection results.
     */
    void postProcess(cv::Mat& features);

    /**
     * @brief Postprocesses the detection results of YOLOv5.
     * 
     * @param features Feature matrix containing raw detection results.
     */
    void postProcessV5(cv::Mat& features);
    
    /**
     * @brief Postprocesses the detection results of YOLOv8.
     * 
     * @param features Feature matrix containing raw detection results.
     */
    void postProcessV8(cv::Mat& features);

    /**
     * @brief Postprocesses the detection results of YOLOv10.
     * 
     * @param features Feature matrix containing raw detection results.
     */
    void postProcessV10(cv::Mat& features);
};

#endif // ENGINE_H