import numpy as np
import cv2
import glob

fs = cv2.FileStorage("../config/camera.yaml", cv2.FILE_STORAGE_READ)
oldMtx = fs.getNode("oldMtx").mat()
newMtx = fs.getNode("newMtx").mat()
dist = fs.getNode("dist").mat()
fs.release() 

images = glob.glob('../media/calib/*.jpg')
for fname in images:
  img = cv2.imread(fname)
  undis = cv2.undistort(img, oldMtx, dist, newMtx)
  
  cv2.imshow("Original vs Undistortion", np.hstack([img, undis]))
  key = cv2.waitKey(0)
cv2.destroyAllWindows()