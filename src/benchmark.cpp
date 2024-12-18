// NOTE:  Latency introduced by camera capture are not yet considered.

#include "engine.hpp"
#include "depth.hpp"

#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;

void loadCalib(const string& path, cv::Mat& oldMtx, cv::Mat& dist, cv::Mat& newMtx) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  // Load matrices
  fs["oldMtx"] >> oldMtx;
  fs["newMtx"] >> newMtx;
  fs["dist"] >> dist;
}

int main() {

  const string configCalib = "../config/calibration.yaml";
  const string configYOLO = "../config/yolo.yaml";
  const string configTask = "../config/task.yaml";
  const string imagePath = "../media/image.jpg"; // use an image acquired by calibrated camera!
  cv::Mat img, undisImg;
  cv::Mat oldMtx, dist, newMtx;

  try {
    img = cv::imread(imagePath);
    if (img.empty()) 
      throw runtime_error("Failed to read image from: " + imagePath);

    // Load calibration matrices
    loadCalib(configCalib, oldMtx, dist, newMtx);

    // Construct objects
    YOLO yolo(configYOLO);
    Estimator estimator(503.800, 503.450, 313.612, 243.104, configTask); // dummy values
    
    // Warm-up
    spdlog::info("Warming-up the network...");
    size_t numIts = 200;
    for (size_t i = 0; i < numIts; ++i) {
      //cv::undistort(img, undisImg, oldMtx, dist, newMtx);
      if(!yolo.infer(img)) {
        spdlog::warn("Inference failed on run #{}. Continuing...", i);
        continue;
      }
      const auto& detections = yolo.getDetections(); // not included in detailed benchmarking
      estimator.computePosition(detections, img.cols);
    }
    // Benchmark
    spdlog::info("Warm-up done. Benchmarking the network...");
    numIts = 2000;
    auto start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numIts; ++i) {
      //cv::undistort(img, undisImg, oldMtx, dist, newMtx);
      if(!yolo.infer(img)) {
        spdlog::warn("Inference failed on run #{}. Continuing...", i);
        continue;
      }
      const auto& detections = yolo.getDetections(); // not included in detailed benchmarking
      estimator.computePosition(detections, img.cols);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration<double, milli>(end - start).count();
    auto avgTime = duration / numIts;
    auto fps = 1000.f / avgTime;
    
    spdlog::info("Benchmarking complete! Displaying results for end-to-end pipeline...");
    spdlog::info("Average latency: {:.2f} ms", avgTime);
    spdlog::info("Average FPS: {:.2f}", fps);
    #ifdef DETAILED_BENCHMARK
      spdlog::warn("This performance is worsened by operation on files.");
      spdlog::warn("Use results in 'log.txt' for precise benchmarking.");
    #endif // DETAILED_BENCHMARK
  } 
  catch (const exception& e) {
    spdlog::error(e.what());
    return 1;
  }
  return 0;
}