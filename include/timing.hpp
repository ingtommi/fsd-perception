#ifndef TIMING_H
#define TIMING_H

#include <unordered_map>
#include <chrono>
#include <fstream>

#ifdef DETAILED_BENCHMARK

  // Declare variables
  #define INIT_TIMER \
    std::chrono::high_resolution_clock::time_point start, end; \
    double duration; \
    std::unordered_map<std::string, double> timings;

  // Start the timer
  #define START_TIMER \
    start = std::chrono::high_resolution_clock::now();

  // Stop the timer, calculate the duration (in ms) and store it for the specific section
  #define END_TIMER(section) \
    end = std::chrono::high_resolution_clock::now(); \
    duration = std::chrono::duration<double, std::milli>(end - start).count(); \
    timings[section] = duration;

  // Log results for all the sections
  #define LOG_TIMER \
    std::ofstream logfile("log.txt", std::ofstream::app); \
    logfile << "Pre-process: " << timings["preprocess"] << ", " \
            << "Inference: " << timings["inference"] << ", " \
            << "Post-process: " << timings["postprocess"] << std::endl;
                             
#else // do nothing
  #define INIT_TIMER
  #define START_TIMER
  #define END_TIMER(section)
  #define LOG_TIMER
#endif // DETAILED_BENCHMARK 
#endif // TIMING_H