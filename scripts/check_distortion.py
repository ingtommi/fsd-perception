import numpy as np
import cv2
import glob

LENS = "6mm"

fs = cv2.FileStorage("../config/cameraCalib_{}.yaml".format(LENS), cv2.FILE_STORAGE_READ)
oldMtx = fs.getNode("oldMtx").mat()
newMtx = fs.getNode("newMtx").mat()
dist = fs.getNode("dist").mat()
fs.release() 

images = glob.glob("../media/calib/{}/*.png".format(LENS))
for fname in images:
  img = cv2.imread(fname)
  undis = cv2.undistort(img, oldMtx, dist, newMtx)
  
  cv2.imshow("Original vs Undistortion", np.hstack([img, undis]))
  key = cv2.waitKey(0)
cv2.destroyAllWindows()