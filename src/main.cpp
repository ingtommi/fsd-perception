#include "engine.hpp"
#include "depth.hpp"
#include "camera.hpp"

#include <stdexcept>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>

using namespace std;

int main() {

  const string configCamCalib = "../config/cameraCalib.yaml";
  const string configCamSetup = "../config/cameraSetup.yaml";
  const string configGeom = "../config/geometry.yaml";
  const string configYOLO = "../config/yolo.yaml";

  try {
    // CONSTRUCT OBJECTS
    Camera camera(configCamCalib, configCamSetup);

    const auto& oldMtx = camera.getOldMtx();
    const auto& dist = camera.getDist();
    const auto& newMtx = camera.getNewMtx();
    // Extract focal length and principal point (multiply by pxSize to convert px to mm)
    const double fx = newMtx.at<double>(0, 0);
    const double fy = newMtx.at<double>(1, 1);
    const double cx = newMtx.at<double>(0, 2);
    const double cy = newMtx.at<double>(1, 2);

    YOLO yolo(configYOLO);

    Estimator estimator(fx, fy, cx, cy, configGeom);

    // RUN ALGORITHM
    cv::Mat frame, undisFrame;
    // Start recording
    camera.start(); // throws exception if fails
    spdlog::info("Started recording...");
    spdlog::info("");

    while (true) { // TODO: consider stopping the loop
      // Capture frame
      if (!camera.capture(frame)) {
        spdlog::warn("Capture failed on current frame. Continuing...");
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

      // Compute and display position of cones related to camera
      const auto& positions = estimator.computePosition(detections, undisFrame.cols);
      if (positions.size() == 0) 
        spdlog::warn("No target detected");
      else {
        for (const auto& position : positions) {
          double x = position.x;
          double z = position.z;
          double distance = sqrt(pow(x, 2) + pow(z, 2));
          spdlog::info("Distance: {}, X: {}, Z: {}", distance, x, z); // just for test
        }
      }
    }
  }
  catch (const exception& e) {
    // Display error
    spdlog::error(e.what());
    return 1;
  }
  return 0;
}