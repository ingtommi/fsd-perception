#ifndef TIMING_H
#define TIMING_H

#include <unordered_map>
#include <chrono>
#include <fstream>

#ifdef DETAILED_BENCHMARK

  // Declare variables
  #define INIT_TIMER_YOLO \
    std::chrono::high_resolution_clock::time_point start, end; \
    double duration; \
    std::unordered_map<std::string, double> timings;

  #define INIT_TIMER_DEPTH \
  std::chrono::high_resolution_clock::time_point start, end; \
  double duration;

  // Start the timer
  #define START_TIMER \
    start = std::chrono::high_resolution_clock::now();

  // Stop the timer, calculate the duration (in ms) and store it for the specific section
  #define END_TIMER_YOLO(section) \
    end = std::chrono::high_resolution_clock::now(); \
    duration = std::chrono::duration<double, std::milli>(end - start).count(); \
    timings[section] = duration;

  // Stop the timer and calculate the duration (in ms)
  #define END_TIMER_DEPTH \
    end = std::chrono::high_resolution_clock::now(); \
    duration = std::chrono::duration<double, std::milli>(end - start).count(); \

  // Log results for all the sections
  #define LOG_TIMER_YOLO \
    std::ofstream logfile("log_yolo.txt", std::ofstream::app); \
    logfile << "Pre-process: " << timings["preprocess"] << ", " \
            << "Inference: " << timings["inference"] << ", " \
            << "Post-process: " << timings["postprocess"] << std::endl;

  #define LOG_TIMER_DEPTH \
    std::ofstream logfile("log_depth.txt", std::ofstream::app); \
    logfile << "Depth: " << duration << std::endl;
                             
#else // do nothing
  #define INIT_TIMER_YOLO
  #define INIT_TIMER_DEPTH
  #define START_TIMER
  #define END_TIMER_YOLO(section)
  #define END_TIMER_DEPTH
  #define LOG_TIMER_YOLO
  #define LOG_TIMER_DEPTH
#endif // DETAILED_BENCHMARK 
#endif // TIMING_H