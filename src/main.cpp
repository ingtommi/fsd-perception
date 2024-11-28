#include "engine.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/videoio.hpp>

using namespace std;

int main() {

  const string configPath = "../config/yolo.yaml";
  const string videoPath = "../media/video.mp4";
  cv::VideoCapture cap;
  cv::VideoWriter writer;

  try {
    // Construct YOLO object
    YOLO yolo(configPath); // throws exceptions
    spdlog::info("YOLO object succesfully created.");

    // Open input video
    cap.open(videoPath); // for camera, pass id instead of path
    if (!cap.isOpened())
      throw runtime_error("Failed to open video from file: " + videoPath); 
    // Get video properties
    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    // Define writer object for saving output video
    writer.open("output.mp4", 
                cv::VideoWriter::fourcc('m','p','4','v'), 
                fps, 
                cv::Size(w, h));
    if (!writer.isOpened())
      throw runtime_error("Failed to open video writer."); 
    // Create frame object
    cv::Mat frame;

    spdlog::info("Starting inference...");
    while (true) {
      // Capture frame
      cap >> frame;
      // Exit if end of capture
      if (frame.empty()) break;

      // Run inference
      if (!yolo.infer(frame)) {
        spdlog::warn("Inference failed on current frame. Continuing...");
        // Continue with next frame
        continue;
      }
      // Draw bounding boxes
      auto detections = yolo.getDetections();
      yolo.drawBbox(frame, detections);
      // Write frame to output video
      writer.write(frame);
    }
    spdlog::info("Inference successfully completed.");
  }
  catch (const exception& e) {
    spdlog::error(e.what());
    cap.release();
    writer.release();
    
    return 1;
  }
  // Release resources
  if (cap.isOpened()) cap.release();
  if (writer.isOpened()) writer.release();

  return 0;
}