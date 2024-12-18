# Process logs generated in detailed benchmarking mode.

import numpy as np

def process_log(filePath, numIts):
  # Containers
  pre_processing_times = []
  inference_times = []
  post_processing_times = []
    
  # Read the file and process the relevant lines
  with open(filePath, 'r') as file:
    # Skip the first numIts lines because it is warm-up
    lines = file.readlines()[numIts:] 
        
    for line in lines:
      # Parse each line to extract the values
      parts = line.split(", ")
      pre_processing = float(parts[0].split(": ")[1])
      inference = float(parts[1].split(": ")[1])
      post_processing = float(parts[2].split(": ")[1])
                
      # Store the values
      pre_processing_times.append(pre_processing)
      inference_times.append(inference)
      post_processing_times.append(post_processing)
    
    # Calculate means and std dev
    pre_mean = np.mean(pre_processing_times)
    pre_dev = np.std(pre_processing_times)
    inference_mean = np.mean(inference_times)
    inference_dev = np.std(inference_times)
    post_mean = np.mean(post_processing_times)
    post_dev = np.std(post_processing_times)
    
    print("Pre-processing: Mean = {:.2f} ms, Std = {:.2f} ms".format(pre_mean, pre_dev))
    print("Inference: Mean = {:.2f} ms, Std = {:.2f} ms".format(inference_mean, inference_dev))
    print("Post-processing: Mean = {:.2f} ms, Std = {:.2f} ms".format(post_mean, post_dev))

# Usage
filePath = "../build/log_yolo.txt"
numIts = 200  # Warm-up iterations
process_log(filePath, numIts)