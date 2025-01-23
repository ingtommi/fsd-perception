#include "depth.hpp"
#include "timing.hpp"

#include <stdexcept>
#include <cmath>
#include <yaml-cpp/yaml.h>

using namespace std;

// Constructor
Estimator::Estimator(const double& fx, const double& fy, const double& cx, const double& cy, const string& configPath) {

  // Set camera parameters, in px
  this->fx = fx;
  this->fy = fy;
  this->cx = cx;
  this->cy = cy;
  // Load object geometry
  if (!setGeometry(configPath))
    throw runtime_error("Failed to load object geometry from: " + configPath);
  // Set thresholds
  this->smallThres = this->fy * this->smallHeight / this->maxDistance;
  this->largeThres = this->fy * this->largeHeight / this->maxDistance;
}

// Compute and return object positions
// This function makes use of macros defined in 'timing.hpp' for benchmarking purpose. Inactive by default.
const vector<Estimator::Position>& Estimator::computePosition(const vector<YOLO::Object>& detections) noexcept {
  
  INIT_TIMER_DEPTH
  START_TIMER

  this->positions.clear(); // NOTE: previous positions are not taken into account because tracking requires global coordinates.
  
  for (const YOLO::Object& detection : detections) {
    
    // Check detection validity
    if (!isValidDetection(detection)) { 
      continue; // skip invalid detections
    }
    // Get data
    float h_bbox = detection.bbox.height;                                          // height of the bounding box (pixels)
    float h_real = (detection.label == 3) ? this->largeHeight : this->smallHeight; // height of the object (meters)
    float x_bbox = detection.bbox.x + (detection.bbox.width / 2);                  // x of the bounding box center (pixels)
    
    // Compute optical distance by inverting pinhole model
    double d_optical = this->fy * h_real / h_bbox;
    // Compute ground distance by using Pythagora
    double d_ground = sqrt(pow(d_optical, 2) - pow(this->cameraHeight, 2));
    // Compute x by using the pinhole model
    double x = d_optical * (x_bbox - this->cx) / this->fx;
    // Compute z by using Pythagora
    double z = sqrt(pow(d_ground, 2) - pow(x, 2));

    this->positions.emplace_back(d_ground, x, z);
  }
  END_TIMER_DEPTH
  LOG_TIMER_DEPTH

  return this->positions;
}

// Check detection validity
bool Estimator::isValidDetection(const YOLO::Object& detection) noexcept {

  // Criteria 1: Cone too far away
  const double thres = (detection.label == 3) ? this->largeThres : this->smallThres;
  if (detection.bbox.height < thres) {
    return false;
  }
  // Criteria 2: Fallen cone
  if (detection.bbox.width > detection.bbox.height) {
    return false;
  }
  return true;
}

// Load object geometry
bool Estimator::setGeometry(const string& configPath) noexcept {
  try {
    // Load file
    YAML::Node config = YAML::LoadFile(configPath);
    // Extract geometry
    if (!config["smallConeHeight"]) return false;
    this->smallHeight = config["smallConeHeight"].as<float>(0.325);

    if (!config["largeConeHeight"]) return false;
    this->largeHeight = config["largeConeHeight"].as<float>(0.505);

    if (!config["cameraHeight"]) return false;
    this->cameraHeight = config["cameraHeight"].as<float>(1.5);

    if (!config["maxDistance"]) return false;
    this->maxDistance = config["maxDistance"].as<float>(20.0);

  } catch (const exception& e) {
    return false;
  }
  return true;
}