import random
import cv2
import numpy as np

#loading original image
image = cv2.imread('Image.jpg')
cv2.imshow("Original", image)

#converting to YUV
image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
src = image.copy()

#adding noise
for i in range(256):
    for j in range(256):
        noise = random.gauss(0, 5)
    	if (src[i, j, 0] + noise <= 255):
    	    src[i, j, 0] += noise
    	else:
    	    src[i, j, 0] -= noise
srcRGB = cv2.cvtColor(src, cv2.COLOR_YCR_CB2RGB)
cv2.imshow("with noise", srcRGB)

#Using Gaussian filter
dst = cv2.GaussianBlur(src, (3, 3), 3)
dst = cv2.cvtColor(dst, cv2.COLOR_YCR_CB2RGB)
dstLaplace = dst.copy()
cv2.imshow("Gaussian", dst)

#Using Laplace operator
dstLaplace = cv2.Laplacian(dstLaplace, 0, 50)
cv2.imshow("Laplacian", dstLaplace)

#waiting ESC key
while True:
    key = cv2.waitKey(1)
    if key == 27: break

