#include "camera.hpp"

#include <spdlog/spdlog.h>

using namespace std;

// Constructor
Camera::Camera(const string& cameraCalib, const string& cameraSetup) {

  // Loads calibration
  if (!loadCalib(cameraCalib))
    throw runtime_error("Failed to load calibration data from: " + cameraCalib);
  // Load setup
  if (!loadSetup(cameraSetup))
    throw runtime_error("Failed to load setup data from: " + cameraSetup);

  int nRet;
  // Initialization
  nRet = is_InitCamera(&this->hCam, NULL);
  if (nRet != IS_SUCCESS)
    throw runtime_error("Camera initialization failed with code: " + to_string(nRet));
  // Set BGR color mode to match OpenCV
  is_SetColorMode(this->hCam, IS_CM_BGR8_PACKED);
  // Enable auto exposure
  double enable = (this->set_autoExp == true) ? 1 : 0; // 1 to enable, 0 to disable
  is_SetAutoParameter(this->hCam, IS_SET_ENABLE_AUTO_SHUTTER, &enable, 0);
  // Set master gain
  if (this->set_gain) {
    nRet = is_SetHardwareGain(this->hCam, this->gain, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER);
    if (nRet != IS_SUCCESS)
      throw runtime_error("Sensor gain setup failed with code: " + to_string(nRet));
  }
  // Set AOI
  if (this->set_aoi) {
    IS_RECT rectAOI;
    rectAOI.s32X = this->aoi_x;
    rectAOI.s32Y = this->aoi_y;
    rectAOI.s32Width = this->aoi_w;
    rectAOI.s32Height = this->aoi_h;
    nRet = is_AOI(this->hCam, IS_AOI_IMAGE_SET_AOI, (void*)&rectAOI, sizeof(rectAOI));
    if (nRet != IS_SUCCESS)
      throw runtime_error("AOI setup failed with code: " + to_string(nRet));
  }
  // Set frame rate
  double newFps = 0; // actual fps value
  if (this->set_fps) {
    nRet = is_SetFrameRate(this->hCam, this->fps, &newFps);
    if (nRet != IS_SUCCESS)
      throw runtime_error("FPS setup failed with code: " + to_string(nRet));
  }
  // Allocate and activate image memory
  nRet = is_AllocImageMem(this->hCam, this->aoi_w, this->aoi_h, 24, &this->pMem, &this->memId);
  if (nRet != IS_SUCCESS) {
    throw runtime_error("Image memory allocation failed with code: " + to_string(nRet));
  }
  nRet = is_SetImageMem(this->hCam, this->pMem, this->memId);
  if (nRet != IS_SUCCESS) {
    throw runtime_error("Image memory activation failed with code: " + to_string(nRet));
  }

  // Show information
  IS_RECT currAOI;
  is_AOI(this->hCam, IS_AOI_IMAGE_GET_AOI, (void*)&currAOI, sizeof(currAOI));
  int gain = is_SetHardwareGain(this->hCam, IS_GET_MASTER_GAIN, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER);
  spdlog::info("----- CAMERA OPTIONS -----");
  spdlog::info("Auto exposure: {}", this->set_autoExp);
  spdlog::info("Gain: {}%", gain);
  spdlog::info("AOI x: {}", currAOI.s32X);
  spdlog::info("AOI y: {}", currAOI.s32Y);
  spdlog::info("AOI width: {}", currAOI.s32Width);
  spdlog::info("AOI height: {}", currAOI.s32Height);
  spdlog::info("FPS: {:.3f} [0 = not set]", newFps);
  spdlog::info("--------------------------");
  spdlog::info("");
}

// Destructor
Camera::~Camera() {
  // Disable the hCam camera handle and release data structures and memory areas
  is_ExitCamera(this->hCam);
}

// Starter
bool Camera::start() noexcept {
  int nRet = is_CaptureVideo(this->hCam, IS_DONT_WAIT); // non-blocking
  if (nRet != IS_SUCCESS)
    return false;
  return true;
}

// Capturer
bool Camera::capture(cv::Mat& frame) noexcept {
  void* pMem_b = nullptr;
  
  // Retrieve the latest frame from the camera memory buffer
  int nRet = is_GetImageMem(hCam, &pMem_b);
  if (nRet != IS_SUCCESS) {
    return false;
  }

  // Convert the image to an OpenCV Mat object
  frame = cv::Mat(this->aoi_h, this->aoi_w, CV_8UC3, pMem_b);
  
  return true;
}

// Getters
const cv::Mat& Camera::getOldMtx() noexcept { return this->oldMtx; }

const cv::Mat& Camera::getDist() noexcept { return this->dist; }

const cv::Mat& Camera::getNewMtx() noexcept { return this->newMtx; }

// Calibration loader
bool Camera::loadCalib(const string& configPath) noexcept {
  try {
    cv::FileStorage fs(configPath, cv::FileStorage::READ);
    // Load matrices
    fs["oldMtx"] >> this->oldMtx;
    fs["newMtx"] >> this->newMtx;
    fs["dist"] >> this->dist;

  } catch (const exception& e) {
    return false;
  }
  return true;
}

// Setup loader
bool Camera::loadSetup(const string& configPath) noexcept {
  try {
    // Load file
    YAML::Node config = YAML::LoadFile(configPath);
    // Exposure
    if (!config["set_autoExp"]) return false;
    this->set_aoi = config["set_autoExp"].as<bool>(false);    
    // AOI
    if (!config["set_aoi"]) return false;
    this->set_aoi = config["set_aoi"].as<bool>(false);
    if(this->set_aoi) {
      if (!config["aoi_x"] || !config["aoi_y"] || !config["aoi_w"] || !config["aoi_h"]) return false;
      this->aoi_x = config["aoi_x"].as<int>(0);
      this->aoi_y = config["aoi_y"].as<int>(0);
      this->aoi_w = config["aoi_w"].as<int>(1920);
      this->aoi_h = config["aoi_h"].as<int>(1200);
    }
    // Gain
    if (!config["set_gain"]) return false;
    this->set_gain = config["set_aoi"].as<bool>(false);
    if (this->set_gain) {
      if (!config["gain"]) return false;
      this->gain = config["gain"].as<int>(50);
    }
    // Frame rate
    if (!config["set_fps"]) return false;
    this->set_fps = config["set_aoi"].as<bool>(false);
    if (this->set_fps) {
      if (!config["fps"]) return false;
      this->fps = config["fps"].as<double>(25.f);
    }

    } catch (const exception& e) {
      return false;
    }
  return true;
}