import cv2
import numpy as np
import glob

# Define image size, checkerboard dimensions (i.e. number of corners) and termination and criteria
IMGSZ = (1920, 1200)
OUTPUT = "./camera_6mm.yaml"
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create arrays to store object points and image points from all the images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Define the world coordinates for 3D points                         #    1     h*w         3
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32) # (batch, points, coordinates)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) # (0,0,0), (1,0,0), ... | z fixed to 0

images = glob.glob('./6mm/*.png')
for fname in images:
  img = cv2.imread(fname)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Find the chessboard corners
  # If desired number of corners are found in the image then ret = true
  ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE) # flags for robustness
  
  if ret == True:
    objpoints.append(objp)
    # Refine pixel coordinates for given 2d points.
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)
    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# Calibrate camera by passing the value of known 3D points (objpoints) 
# and corresponding pixel coordinates of the detected corners (imgpoints)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Refine camera matrix using alpha=1 to retain all pixels
newMtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, IMGSZ, 1, IMGSZ)

# Compute re-projection error
mean_error = 0
for i in range(len(objpoints)):
  imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
  error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
  mean_error += error
 
print("Total error: {}".format(mean_error/len(objpoints)))

# Save camera intrinisc matrix and distortion coefficients
fs = cv2.FileStorage(OUTPUT, cv2.FILE_STORAGE_WRITE)
fs.write("oldMtx", mtx)
fs.write("newMtx", newMtx)
fs.write("dist", dist)
fs.release()