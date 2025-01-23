#include "engine.hpp"
#include "depth.hpp"
#include "camera.hpp"

#include <stdexcept>
#include <cmath>
#include <opencv2/calib3d.hpp>

using namespace std; 

int main() {

  const string configCamCalib = "../config/cameraCalib.yaml";
  const string configCamSetup = "../config/cameraSetup.yaml";
  const string configGeom = "../config/configGeom.yaml";
  const string configYOLO = "../config/configYOLO.yaml";
  //const string imagePath = "../media/6mm.png";

  try {
    // CONSTRUCT OBJECTS
    Camera camera(configCamCalib, configCamSetup);
    
    const auto& aoi = camera.getAOI();
    const cv::Size size(aoi.w, aoi.h);

    const auto& calib = camera.getCalib();
    const cv::Mat& oldMtx = calib.oldMtx;
    const cv::Mat& dist = calib.dist;
    const cv::Mat& newMtx = calib.newMtx;
    // Extract focal length and principal point (multiply by pxSize if converting px to mm is needed)
    const double fx = newMtx.at<double>(0, 0);
    const double fy = newMtx.at<double>(1, 1);
    const double cx = newMtx.at<double>(0, 2);
    const double cy = newMtx.at<double>(1, 2);

    YOLO yolo(configYOLO);
    Estimator estimator(fx, fy, cx, cy, configGeom);

    cv::Mat map1, map2;
    cv::Mat frame, undisFrame;
    
    // RUN ALGORITHM
    
    // Compute undistortion maps
    cv::initUndistortRectifyMap(oldMtx, dist, cv::Mat(), newMtx, size, CV_32FC1, map1, map2);
    
    // Start recording
    camera.start(); // throws exception if fails
    spdlog::info("Started recording...");
    spdlog::info("");

    while (true) {
      // Capture frame
      if (!camera.capture(frame)) {
        spdlog::warn("Capture failed on current frame. Continuing...");
        // Continue with next frame
        continue;
      }

      //frame = cv::imread(imagePath); // alternative to use pre-acquired image

      // Remap frame (undistortion)
      cv::remap(frame, undisFrame, map1, map2, cv::INTER_LINEAR);

      // Run inference and get detections
      if (!yolo.infer(undisFrame)) {
        spdlog::warn("Inference failed on current frame. Continuing...");
        // Continue with next frame
        continue;
      }
      const auto& detections = yolo.getDetections();
  
      // Compute and display position of cones related to camera
      const auto& positions = estimator.computePosition(detections);
      if (positions.size() == 0) 
        spdlog::warn("No target detected");
      else {
        for (const auto& position : positions) {
          double dist = position.dist;
          double x = position.x;
          double z = position.z;
          spdlog::info("Distance: {:.3f} m (X: {:.3f}, Z: {:.3f})", dist, x, z); // just for test
        }
        spdlog::info("");
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