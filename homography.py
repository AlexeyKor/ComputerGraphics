import cv2
import numpy as np

#loading image
image = cv2.imread('matmex.jpg')
cv2.imshow("Original", image)

#Computing homogeneous matrix
src = np.array([[298, 475], [460, 475], [298, 70], [460, 70]], np.float32)
dst = np.array([[298, 475], [510, 475], [322, 158], [490, 153]], np.float32)
HMatrix = cv2.getPerspectiveTransform(src, dst)

#Removing the projective distortion
warpImage = cv2.warpPerspective(image, np.linalg.inv(HMatrix), (800, 600))
cv2.imshow("Warp Image", warpImage)

#waiting ESC key
while True:
    key = cv2.waitKey(1)
    if key == 27: break
