# Process logs generated in detailed benchmarking mode.

import numpy as np

def process_log(filePath, numIts):
  # Container
  depth_times = []
    
  # Read the file and process the relevant lines
  with open(filePath, 'r') as file:
    # Skip the first numIts lines because it is warm-up
    lines = file.readlines()[numIts:] 
        
    for line in lines:
      # Parse each line to extract the values
      time = float(line.split(": ")[1])     
      # Store the value
      depth_times.append(time)
    
    # Calculate means and std dev
    depth_mean = np.mean(depth_times)
    depth_dev = np.std(depth_times)
    
    print("Depth estimation: Mean = {:.2f} ms, Std = {:.2f} ms".format(depth_mean, depth_dev))

# Usage
filePath = "../build/log_depth.txt"
numIts = 200  # Warm-up iterations
process_log(filePath, numIts)