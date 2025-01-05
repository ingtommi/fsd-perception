// NOTE:  Latency introduced by camera capture are not yet considered.

#include "engine.hpp"
#include "depth.hpp"

#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;

/*
void drawBBox(cv::Mat& frame, const vector<Object>& detections) noexcept {
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
*/

void loadCalib(const string& path, cv::Mat& oldMtx, cv::Mat& dist, cv::Mat& newMtx) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  // Load matrices
  fs["oldMtx"] >> oldMtx;
  fs["newMtx"] >> newMtx;
  fs["dist"] >> dist;
}

void runLoop(YOLO& yolo, Estimator& estimator, cv::Mat& img, cv::Mat& undisImg, cv::Mat& oldMtx, cv::Mat& newMtx, cv::Mat& dist, size_t numIts) {
  for (size_t i = 0; i < numIts; ++i) {
    cv::undistort(img, undisImg, oldMtx, dist, newMtx);
    if(!yolo.infer(undisImg)) {
      spdlog::warn("Inference failed on run #{}. Continuing...", i);
      continue;
    }
    const auto& detections = yolo.getDetections(); // not included in detailed benchmarking
    estimator.computePosition(detections, img.cols);
  }
}

int main() {

  const string configCalib = "../config/cameraCalib.yaml";
  const string configYOLO = "../config/yolo.yaml";
  const string configGeom = "../config/geometry.yaml";
  const string imagePath = "../media/image.jpg"; // use an image acquired by calibrated camera!
  cv::Mat img;// undisImg;
  cv::Mat oldMtx, dist, newMtx;

  try {
    img = cv::imread(imagePath);
    if (img.empty()) 
      throw runtime_error("Failed to read image from: " + imagePath);

    // Load calibration matrices
    loadCalib(configCalib, oldMtx, dist, newMtx);
    const double fx = newMtx.at<double>(0, 0);
    const double fy = newMtx.at<double>(1, 1);
    const double cx = newMtx.at<double>(0, 2);
    const double cy = newMtx.at<double>(1, 2);

    // Construct objects
    YOLO yolo(configYOLO);
    Estimator estimator(fx, fy, cx, cy, configGeom);
    
    // Warm-up
    spdlog::info("Warming-up the network...");
    runLoop(yolo, estimator, img, undisImg, oldMtx, newMtx, dist, 200); // run for 200 iterations
    
    // Benchmark
    spdlog::info("Warm-up done. Benchmarking the network...");
    auto start = chrono::high_resolution_clock::now();

    runLoop(yolo, estimator, img, undisImg, oldMtx, newMtx, dist, 2000); // run for 2000 iterations
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration<double, milli>(end - start).count();
    auto avgTime = duration / 2000;
    auto fps = 1000.f / avgTime;
    
    spdlog::info("Benchmarking complete! Displaying results for end-to-end pipeline...");
    spdlog::info("Average latency: {:.2f} ms", avgTime);
    spdlog::info("Average FPS: {:.2f}", fps);
    #ifdef DETAILED_BENCHMARK
      spdlog::warn("This performance is worsened by operation on files.");
      spdlog::warn("Use results in 'log_yolo.txt' and 'log_depth.txt' for precise benchmarking.");
    #endif // DETAILED_BENCHMARK
  } 
  catch (const exception& e) {
    spdlog::error(e.what());
    return 1;
  }
  return 0;
}