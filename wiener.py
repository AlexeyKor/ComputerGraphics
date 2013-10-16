import cv2
import numpy as np

#loading image
image = cv2.imread('wiener-problem')
cv2.imshow("Original", image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#Furrier transform
dst = np.fft.fft2(image)
dst = np.fft.fftshift(dst)

#Creating kernel for degradation function
h = np.zeros((25, 25))

h[12][12] = 1
h[12][13] = 0.5
h[12][14] = 0.5
h[12][15] = 0.5
h[12][16] = 0.4
h[12][17] = 0.4
h[12][18] = 0.4
h[12][19] = 0.3
h[12][20] = 0.3
h[12][21] = 0.3
h[12][20] = 0.2
h[12][23] = 0.2
h[12][24] = 0.2

#Using Wiener filter
H = np.fft.fft2(h, [256, 384])
H = np.fft.fftshift(H)
dst = (np.conj(H) / (np.conj(H) * H + 0.001)) * dst

#inverse furrier transform
dst = np.fft.ifftshift(dst)
dst = np.fft.ifft2(dst)

#converting to opencv array format from numpy array format
final = image.copy()
for i in range(256):
    for j in range(384):
	final[i][j] = np.abs(dst[i][j])

cv2.imshow("Result", final)
cv2.imwrite("result.bmp", final)

#Creating kernel for sharpening and adding brightness
sharpKernel = np.zeros((3,3))

sharpKernel[0][0]= -0.1
sharpKernel[0][1]= -0.1
sharpKernel[0][2]= -0.1
sharpKernel[2][0]= -0.1
sharpKernel[2][1]= -0.1
sharpKernel[2][2]= -0.1
sharpKernel[1][0]= -0.1
sharpKernel[1][2]= -0.1
sharpKernel[1][1]= 4

#Sharpening and adding brightness
final = cv2.filter2D(final,-1, sharpKernel)

cv2.imshow("Sharp Result", final)
cv2.imwrite("sharpResult.bmp", final)


#waiting ESC key
while True:
    key = cv2.waitKey(1)
    if key == 27: break
