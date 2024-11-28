#include "engine.hpp"
#include <spdlog/spdlog.h>
#include <chrono>
#include <opencv2/videoio.hpp>

using namespace std;

int main() {

  const string configPath = "../config/yolo.yaml";
  const string imagePath = "../media/image.jpg";

  try {
    // Construct YOLO object
    YOLO yolo(configPath); // throws exceptions

    auto img = cv::imread(imagePath);
    if (img.empty()) 
      throw runtime_error("Failed to read image from: " + imagePath);
    
    // Warm-up
    spdlog::info("Warming-up the network...");
    size_t numIts = 200;
    for (size_t i = 0; i < numIts; ++i) {
      if(!yolo.infer(img)) {
        spdlog::warn("Inference failed on run #{}. Continuing...", i);
        continue;
      }
    }
    // Benchmark
    spdlog::info("Warm-up done. Benchmarking the network...");
    numIts = 2000;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numIts; ++i) {
      if(!yolo.infer(img)) {
        spdlog::warn("Inference failed on run #{}. Continuing...", i);
        continue;
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    auto avgTime = duration / numIts;
    auto fps = 1000.f / avgTime;
    
    spdlog::info("Benchmarking complete! Displaying results for end-to-end pipeline...");
    spdlog::info("Average latency: {:.2f} ms", avgTime);
    spdlog::info("Average FPS: {:.2f}", fps);
    #ifdef DETAILED_BENCHMARK
      spdlog::warn("This performance is worsened by operations on file.");
      spdlog::warn("Use results in 'log.txt' for precise benchmarking.");
    #endif // DETAILED_BENCHMARK
  } 
  catch (const exception& e) {
    spdlog::error(e.what());
    return 1;
  }
  return 0;
}