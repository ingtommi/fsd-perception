#ifndef CAMERA_H
#define CAMERA_H

#include "ueye.h"
#include "opencv2/opencv.hpp"
#include <yaml-cpp/yaml.h>

class Camera {
  public:
    /**
     * @brief Construct a Camera object from camera calibration and setup parameters.
     * 
     * @param cameraCalib Path to the file containing calibration data.
     * @param cameraSetup Path to the file containing custom setup.
     */
    Camera(const std::string& cameraCalib, const std::string& cameraSetup);

    /** 
     * @brief Destructs a Camera object.
     */
    ~Camera();
    
    /**
     * @brief Starts recording by activating the camera's live video mode.
     * 
     */
    void start();
    
    /**
     * @brief Retrieves the latest frame from image memory.
     * 
     * @param frame OpenCV Mat where to store the capture. 
     * @return True if successful, False otherwise.
     */
    bool capture(cv::Mat& frame) noexcept;
    
    /**
     * @brief Get the camera intrinsic matrix.
     * 
     * @return Reference to the matrix.
     */
    const cv::Mat& getOldMtx() noexcept;

    /**
     * @brief Get the camera distortion matrix.
     * 
     * @return Reference to the matrix.
     */
    const cv::Mat& getDist() noexcept;

    /**
     * @brief Get the refined camera intrinsic matrix.
     * 
     * @return Reference to the matrix.
     */
    const cv::Mat& getNewMtx() noexcept;

  private:

    HIDS hCam = 0;        // camera handle set to 0 to take first camera available
    char* pMem = nullptr; // pointer to the memory to be allocated
    int memId = 0;        // id of this memory
  
    // Calibration parameters
    cv::Mat oldMtx;       // Matrix containing camera intrinsic parameters.
    cv::Mat newMtx;       // Matrix containing refined camera intrinsic parameters.
    cv::Mat dist;         // Matrix containing camera distortion parameters.

    // Setup parameters
    bool auto_exp;        // auto-exposure on/off
    int gain;             // percentage of the maximum gain.
    int aoi_x;            // x-coordinate of AOI in px.
    int aoi_y;            // y-coordinate of AOI in px.
    int aoi_w;            // width of the AOI in px.
    int aoi_h;            // height of the AOI in px.
    double fps;           // fps value to be used.
    double actual_fps;    // fps value actually set

    /**
     * @brief Loads camera intrinsic parameters from calibration file.
     * 
     * @param configPath Path to the file containing camera calibration paramaters.
     * @return True if successful, False otherwise. 
     */
    bool loadCalib(const std::string& configPath) noexcept;

    /**
     * @brief Loads sensor setup parameters from setup file.
     * 
     * @param configPath Path to the file containing camera setup paramaters.
     * @return True if successful, False otherwise. 
     */
    bool loadSetup(const std::string& configPath) noexcept;
};

#endif // CAMERA_H