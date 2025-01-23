// IDS error codes at: https://www.1stvision.com/cameras/IDS/IDS-manuals/uEye_Manual/sdk_fehlermeldungen.html

#include "camera.hpp"

#include <spdlog/spdlog.h>

using namespace std;

// Constructor
Camera::Camera(const string& cameraCalib, const string& cameraSetup) {

  // Loads calibration
  if (!loadCalib(cameraCalib))
    throw runtime_error("Failed to load camera calibration data from: " + cameraCalib);
  // Load setup
  if (!loadSetup(cameraSetup))
    throw runtime_error("Failed to load camera setup data from: " + cameraSetup);

  int nRet;
  // Initialization
  nRet = is_InitCamera(&this->hCam, NULL);
  if (nRet != IS_SUCCESS)
    throw runtime_error("Failed to initialize camera. Error code: " + to_string(nRet));

  // Set BGR color mode to match OpenCV
  nRet = is_SetColorMode(this->hCam, IS_CM_BGR8_PACKED);
  if (nRet != IS_SUCCESS)
    throw runtime_error("Failed to set camera color mode. Error code: " + to_string(nRet));

  // Enable auto exposure
  double enable = this->auto_exp ? 1 : 0; // 1 to enable, 0 to disable
  nRet = is_SetAutoParameter(this->hCam, IS_SET_ENABLE_AUTO_SHUTTER, &enable, 0);
  if (nRet != IS_SUCCESS)
    throw runtime_error("Failed to set camera auto exposure. Error code: " + to_string(nRet));

  // Set master gain
  nRet = is_SetHardwareGain(this->hCam, this->gain, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER);
  if (nRet != IS_SUCCESS)
    throw runtime_error("Failed to set camera gain. Error code: " + to_string(nRet));

  // Set AOI
  IS_RECT rectAOI;
  rectAOI.s32X = this->aoi.x;
  rectAOI.s32Y = this->aoi.y;
  rectAOI.s32Width = this->aoi.w;
  rectAOI.s32Height = this->aoi.h;
  nRet = is_AOI(this->hCam, IS_AOI_IMAGE_SET_AOI, (void*)&rectAOI, sizeof(rectAOI));
  if (nRet != IS_SUCCESS)
    throw runtime_error("Failed to set camera AOI. Error code: " + to_string(nRet));

  // Set frame rate
  nRet = is_SetFrameRate(this->hCam, this->fps, &this->actual_fps);
  if (nRet != IS_SUCCESS)
    throw runtime_error("Failed to set camera FPS. Error code: " + to_string(nRet));

  // Allocate and activate image memory
  nRet = is_AllocImageMem(this->hCam, this->aoi.w, this->aoi.h, 24, &this->pMem, &this->memId);
  if (nRet != IS_SUCCESS)
    throw runtime_error("Failed to allocate image memory. Error code: " + to_string(nRet));
  nRet = is_SetImageMem(this->hCam, this->pMem, this->memId);
  if (nRet != IS_SUCCESS)
    throw runtime_error("Failed to activate image memory. Error code: " + to_string(nRet));

  // Show information
  spdlog::info("----- CAMERA OPTIONS -----");
  spdlog::info("Auto exposure: {}", this->auto_exp);
  spdlog::info("Gain: {}%", this->gain);
  spdlog::info("AOI x: {}", this->aoi.x);
  spdlog::info("AOI y: {}", this->aoi.y);
  spdlog::info("AOI width: {}", this->aoi.w);
  spdlog::info("AOI height: {}", this->aoi.h);
  spdlog::info("FPS: {:.2f}", this->actual_fps);
  spdlog::info("--------------------------");
  spdlog::info("");
}

// Destructor
Camera::~Camera() {
  // Disable the hCam camera handle and release data structures and memory areas
  is_ExitCamera(this->hCam);
}

// Starter
void Camera::start() {
  int nRet = is_CaptureVideo(this->hCam, IS_DONT_WAIT); // non-blocking
  if (nRet != IS_SUCCESS)
    throw runtime_error("Failed to start recording. Error code: " + to_string(nRet));
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
  frame = cv::Mat(this->aoi.h, this->aoi.w, CV_8UC3, pMem_b);
  
  return true;
}

// Getters
const Camera::Calib& Camera::getCalib() noexcept { return this->calib; }

const Camera::AOI& Camera::getAOI() noexcept { return this->aoi; }

// Calibration loader
bool Camera::loadCalib(const string& configPath) noexcept {
  try {
    cv::FileStorage fs(configPath, cv::FileStorage::READ);
    // Load matrices
    fs["oldMtx"] >> this->calib.oldMtx;
    fs["newMtx"] >> this->calib.newMtx;
    fs["dist"] >> this->calib.dist;

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
    // Auto exposure
    if (!config["auto_exp"]) return false;
    this->auto_exp = config["auto_exp"].as<bool>(true);    
    // Gain
    if (!config["gain"]) return false;
    this->gain = config["gain"].as<int>(50);
    // AOI
    if (!config["aoi_x"] || !config["aoi_y"] || !config["aoi_w"] || !config["aoi_h"]) return false;
    this->aoi.x = config["aoi_x"].as<int>(0);
    this->aoi.y = config["aoi_y"].as<int>(0);
    this->aoi.w = config["aoi_w"].as<int>(1920);
    this->aoi.h = config["aoi_h"].as<int>(1200);
    // Frame rate
    if (!config["fps"]) return false;
    this->fps = config["fps"].as<double>(20.f);
  } catch (const exception& e) {
    return false;
  }
  return true;
}