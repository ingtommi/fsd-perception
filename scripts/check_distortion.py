import numpy as np
import cv2
import glob

LENS = "8mm"
IMGSZ = (1920, 1200)

fs = cv2.FileStorage(f"../config/{LENS}_{IMGSZ[0]}x{IMGSZ[1]}.yaml", cv2.FILE_STORAGE_READ)
oldMtx = fs.getNode("oldMtx").mat()
newMtx = fs.getNode("newMtx").mat()
dist = fs.getNode("dist").mat()
fs.release() 

path = "../media/8mm_full.png"
img = cv2.imread(path)
#undis = cv2.undistort(img, oldMtx, dist, newMtx)
map1, map2 = cv2.initUndistortRectifyMap(oldMtx, dist, None, newMtx, (img.shape[1], img.shape[0]), cv2.CV_32FC1)
undis = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

cv2.imwrite("difference.png", np.hstack([img, undis]))