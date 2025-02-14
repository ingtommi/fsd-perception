import matplotlib.pyplot as plt
import numpy as np

# Ground truth
X_gt = np.array([-1.05, 0.93, 0])
Z_gt = np.array([4, 6, 9])

############### 8mm ###############

### NANO, 1920x1200 ###
#X_meas = np.array([-1.160, 0.566, -0.52]) # <-- alpha=0 (conf = 0.3, eppure il modello li riconosce con conf > 0.8...)
#Z_meas = np.array([3.018, 5.266, 8.405])

### SMALL, 1920x1200 ###
#X_meas = np.array([-1.143, 0.570, -0.543])
#Z_meas = np.array([2.945, 5.382, 8.68])

############### 6mm ###############

# nano, 1920x1200
#X_meas = np.array([-1.165, 0.579, -0.496])
#Z_meas = np.array([3.110, 5.077, 8.696])

# small, 1920x1200
X_meas = np.array([-1.158, 0.614, -0.477])
Z_meas = np.array([3.075, 5.455, 8.510])

###################################

# ERROR
distances = np.sqrt((X_gt - X_meas)**2 + (Z_gt - Z_meas)**2)

# Metrics
mean_error = np.mean(distances)       # Mean error
rmse = np.sqrt(np.mean(distances**2)) # Root Mean Square Error
max_error = np.max(distances)         # Maximum error

print(f"Max Error: {max_error:.3f} m")
print(f"Mean Error: {mean_error:.2f} m")
print(f"RMSE: {rmse:.3f} m")

# PLOT
plt.figure(figsize=(6, 6))

# Plot ground truth cones
plt.scatter(X_gt, Z_gt, color='red', label='Ground Truth Cones')

# Plot measurement cones
plt.scatter(X_meas, Z_meas, color='blue', label='Estimated Cones')

# Plot the vehicle position
plt.scatter(0, 0, color='black', label='Camera', zorder=5)

# Axes and labels
plt.axhline(0, color='black', linewidth=0.5)  # Forward direction
plt.axvline(0, color='black', linewidth=0.5)  # Camera center
plt.xlabel("Lateral Distance (m)")
plt.ylabel("Forward Distance (m)")
plt.title("Bird's-Eye View Map")
plt.legend(fontsize=12)
plt.grid(True)
plt.show()