import cv2
import numpy as np
import math

#loading original image and converting to grayscale
image = cv2.imread('Image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.imshow("Original in grayscale", image)

#Furrier transform
dst = np.fft.fft2(image)
dst = np.fft.fftshift(dst)
dstIdeal = np.copy(dst)
dstButter = np.copy(dst)
BHPF = np.copy(dst)

#Ideal high pass filter
for i in range(98, 158):
    for j in range(98, 158):
        dstIdeal[i][j] = 0;
#inverse furrier transform
dstIdeal = np.fft.ifftshift(dstIdeal)
dstIdeal = np.fft.ifft2(dstIdeal)

#counting H(u,v) for butterworth filter
for i in range(256):
    for j in range(256):
	if (i != 128 or j != 128):
	        BHPF[i][j] = 1 / (1 + math.pow(30 / math.sqrt(math.pow(127 - (i-1), 2) + math.pow(127 - (j-1), 2)), 4))
BHPF[128][128] = 0

#using butterworth filter
dstButter = BHPF * dstButter

#inverse furrier transform
dstButter = np.fft.ifftshift(dstButter)
dstButter = np.fft.ifft2(dstButter)

#loops for showing real and imaginary parts
#final = image.copy()
#for i in range(256):
#    for j in range(256):
#	final[i][j] = np.real(dstIdeal[i][j])
#cv2.imshow("Result real", final)

#final = image.copy();
#for i in range(256):
#    for j in range(256):
#	final[i][j] = np.imag(dstIdeal[i][j])
#cv2.imshow("Result imaginary", final)

#converting to opencv array format from numpy array format
finalIdeal = image.copy()
for i in range(256):
    for j in range(256):
	finalIdeal[i][j] = np.abs(dstIdeal[i][j])
cv2.imshow("Result Ideal filter", finalIdeal)

#converting to opencv array format from numpy array format
finalButter = image.copy()
for i in range(256):
    for j in range(256):
	finalButter[i][j] = np.abs(dstButter[i][j])
cv2.imshow("Result Butter filter", finalButter)


#waiting ESC key
while True:
    key = cv2.waitKey(1)
    if key == 27: break
