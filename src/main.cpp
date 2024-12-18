#include "engine.hpp"
#include "depth.hpp"
#include "camera.hpp"

#include <stdexcept>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>

using namespace std;

/**
 * @brief Draws bounding boxes.
 * 
 * @param frame Frame where to draw bounding boxes.
 * @param detections Vector of detected objects.
 */
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

int main() {

  const string configCamCalib = "../config/calibration.yaml";
  const string configCamSetup = "../config/camera.yaml";
  const string configTask = "../config/task.yaml";
  const string configYOLO = "../config/yolo.yaml";

  try {
    // CONSTRUCT OBJECTS
    Camera camera(configCamCalib, configCamSetup);

    const auto& oldMtx = camera.getOldMtx();
    const auto& dist = camera.getDist();
    const auto& newMtx = camera.getNewMtx();
    // Extract focal length and principal point (multipy by pxSize to convert px to mm)
    const double& fx = newMtx.at<double>(0, 0);
    const double& fy = newMtx.at<double>(1, 1);
    const double& cx = newMtx.at<double>(0, 2);
    const double& cy = newMtx.at<double>(1, 2);

    YOLO yolo(configYOLO);

    Estimator estimator(fx, fy, cx, cy, configTask);

    // RUN ALGORITHM
    cv::Mat frame, undisFrame;
    // Start recording
    if (!camera.start())
      throw runtime_error("Failed to start recording.");
    spdlog::info("Started recording...");
    spdlog::info("");

    while (true) { // TODO: consider stopping the loop
      // Capture frame
      if (!camera.capture(frame)) {
        spdlog::warn("Captured failed on current frame. Continuing...");
        // Continue with next frame
        continue;
      }

      // Undistort frame
      cv::undistort(frame, undisFrame, oldMtx, dist, newMtx);

      // Run inference and get detections
      if (!yolo.infer(undisFrame)) {
        spdlog::warn("Inference failed on current frame. Continuing...");
        // Continue with next frame
        continue;
      }
      const auto& detections = yolo.getDetections();

      // Compute and get position of cones related to camera
      const auto& positions = estimator.computePosition(detections, undisFrame.cols);
      spdlog::info("Number of positions: ", positions.size()); // just for test
    }
  }
  catch (const exception& e) {
    // Display error
    spdlog::error(e.what());
    return 1;
  }
  return 0;
}