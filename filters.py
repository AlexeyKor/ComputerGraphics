import random
import cv2
import numpy as np

#loading original image
image = cv2.imread('Image.jpg')
cv2.imshow("Original", image)

#converting to YUV
src = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
#src = image.copy()
#for i in range(256):
#    for j in range(256):
#        src[i, j, 0] = 0.299 * image[i, j, 0] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 2]
#        src[i, j, 1] = -0.14713 * image[i, j, 0] - 0.28886 * image[i, j, 1] + 0.436 * image[i, j, 2] + 128
#        src[i, j, 2] = 0.615 * image[i, j, 0] - 0.51499 * image[i, j, 1] - 0.10001 * image[i, j, 2] + 128

#adding noise
for i in range(256):
    for j in range(256):
        src[i, j, 0] += random.gauss(0, 5)
srcRGB = cv2.cvtColor(src, cv2.COLOR_YCR_CB2RGB)
cv2.imshow("with noise", srcRGB)

#Using Gaussian filter
dst1 = cv2.GaussianBlur(src, (3, 3), 3)
dst1 = cv2.cvtColor(dst1, cv2.COLOR_YCR_CB2RGB)
cv2.imshow("Gaussian", dst1)

#Showing Gaussian difference
dst1Diff = dst1.copy()
for i in range(256):
    for j in range(256):
        for k in range(3):
            if(dst1Diff[i, j, k] < src[i, j, k]):
                dst1Diff[i, j, k] = src[i, j, k] - dst1Diff[i, j, k]
            else:
                dst1Diff[i, j, k] = dst1Diff[i, j, k] - src[i, j, k]
dst1Diff = cv2.cvtColor(dst1Diff, cv2.COLOR_YCR_CB2RGB)
cv2.imshow("Gaussian difference", dst1Diff)

#Using Bilateral filter
dst2 = cv2.bilateralFilter(src, 5, 100, 100) 
dst2 = cv2.cvtColor(dst2, cv2.COLOR_YCR_CB2RGB)
cv2.imshow("Bilateral", dst2)

#Showing Bilateral difference
dst2Diff = dst2.copy()
for i in range(256):
    for j in range(256):
        for k in range(3):
            if(dst2Diff[i, j, k] < src[i, j, k]):
                dst2Diff[i, j, k] = src[i, j, k] - dst2Diff[i, j, k]
            else:
                dst2Diff[i, j, k] = dst2Diff[i, j, k] - src[i, j, k]
dst2Diff = cv2.cvtColor(dst2Diff, cv2.COLOR_YCR_CB2RGB) 
cv2.imshow("Bilateral difference", dst2Diff)

#Using NLM filter
dst3 = cv2.fastNlMeansDenoisingColored(src)
dst3 = cv2.cvtColor(dst3, cv2.COLOR_YCR_CB2RGB)
cv2.imshow("NLM", dst3)

#Showing NLM difference
dst3Diff = dst3.copy()
for i in range(256):
    for j in range(256):
        for k in range(3):
            if(dst3Diff[i, j, k] < src[i, j, k]):
                dst3Diff[i, j, k] = src[i, j, k] - dst3Diff[i, j, k]
            else:
                dst3Diff[i, j, k] = dst3Diff[i, j, k] - src[i, j, k]
dst3Diff = cv2.cvtColor(dst3Diff, cv2.COLOR_YCR_CB2RGB)
cv2.imshow("NLM difference", dst3Diff)

#waiting ESC key
while True:
    key = cv2.waitKey(1)
    if key == 27: break

