# Calculate mean and stddev values of custom dataset to apply Z-Score Normalization.

# FSOCO-split: - Mean (BGR): [0.495, 0.511, 0.475]
#              - StdDev (BGR): [0.200, 0.192, 0.193]

import cv2
import numpy as np
import os
from tqdm import tqdm

# Path to your dataset images
dataset_path = "/path/to/train/images"

# Initialize accumulators for mean and standard deviation
num_images = 0
mean_accum = np.zeros(3)
std_accum = np.zeros(3)

# Loop through all images in the dataset
for filename in tqdm(os.listdir(dataset_path)):
    # Load image (BGR by default)
    img_path = os.path.join(dataset_path, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue  # Skip any non-image files
    
    # Convert to float32 and scale by 255
    img = img.astype(np.float32) / 255.0
    
    # Compute mean and standard deviation for this image
    mean, std_dev = cv2.meanStdDev(img)
    
    # Accumulate the means and standard deviations
    mean_accum += mean.flatten()
    std_accum += std_dev.flatten()
    num_images += 1

# Compute dataset-wide mean and standard deviation
mean_dataset = mean_accum / num_images
std_dataset = std_accum / num_images

print("Mean of dataset (BGR): ", mean_dataset)
print("Standard Deviation of dataset (BGR): ", std_dataset)