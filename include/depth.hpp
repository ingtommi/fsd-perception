#ifndef DEPTH_H
#define DEPTH_H

#include "engine.hpp"

#include <opencv2/core.hpp>

/**
 * @brief Depth estimation class.
 * 
 * This class performs depth estimation using the concept of triangle similarity.
 */
class Estimator {
  public:
    /**
     * @brief Struct to hold position estimates.
     *
     * This struct holds the estimated x and y coordinates of objects.
     */
    struct Position {
      const double dist;
      const double x;
      const double z;

      /**
       * @brief Constructs a Position object.
       *
       * @param dist Distance of the object.
       * @param x x-coordinate of the object.
       * @param z z-coordinate of the object.
       */
      Position(double& dist, double& x, double& z) : dist(dist), x(x), z(z) {}
    };

    /**
     * @brief Constructs a DepthEstimator object from camera parameters and object geometry file.
     * 
     * @param fx Focal length along the x-axis in px.
     * @param fy Focal length along the y-axis in px.
     * @param cx X-coordinate of the principal point in px.
     * @param cy Y-coordinate of the principal point in px.
     * @param configPath Path to the file containing object geometry data.
     */
    Estimator(const double& fx, const double& fy, const double& cx, const double& cy, const std::string& configPath);

    /**
     * @brief Compute and return the positions of cones related to the camera.
     * 
     * @param detections Vector of detected objects.
     * @return Reference to vector of estimated positions.
     */
    const std::vector<Position>& computePosition(const std::vector<YOLO::Object>& detections) noexcept;

  private:    
    // Camera parameters
    double fx;                 // Focal length of the camera along x-axis in px.
    double fy;                 // Focal length of the camera along y-axis in px.
    double cx;                 // x-coordinate of the camera principal point in px.
    double cy;                 // y-coordinate of the camera principal point in px.

    // Geometry
    float smallHeight;         // Height in meters of the large orange cones, from FS rules.
    float largeHeight;         // Height in meters of the yellow, blue and small orange cones, from FS rules.
    float cameraHeight;        // Height in meters of the camera from ground.
    float maxDistance;         // Max distance in meters to consider a cone.

    double smallThres;         // Threshold in pixels for small cones based on maxDistance.
    double largeThres;         // Threshold in pixels for large cones based on maxDistance.

    // Results
    std::vector<Position> positions; // Vector of estimated positions.
 
    /**
     * @brief Checks whether a YOLO detection is to be used for distance estimation.
     * 
     * Out of all the detections, some must be discarded because the object may have fallen, 
     * or it is too far away to be of interest in the current frame.
     * 
     * @param detection Detected object.
     * @return True if the detection is to be used, False otherwise.
     */
    bool isValidDetection(const YOLO::Object& detection) noexcept;

    /**
     * @brief Loads object geometry from config path.
     * 
     * @param configPath Path to the file containing object geometry data.
     * @return True if successful, False otherwise.
     */
    bool setGeometry(const std::string& configPath) noexcept;
};

#endif // DEPTH_H