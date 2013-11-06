import cv2
import numpy as np

# Loading images and converting to grayscale
source = cv2.imread('box.jpg')
sourceRotated = cv2.imread('box-rotated.jpg')
gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
grayRotated = cv2.cvtColor(sourceRotated, cv2.COLOR_BGR2GRAY)

# Creating SIFT object
sift = cv2.SIFT()

# Detecting keypoints and computing descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(grayRotated, None)

# Showing images with keypoints
source = cv2.drawKeypoints(gray, keypoints1, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Original keypoints', source)
sourceRotated = cv2.drawKeypoints(grayRotated, keypoints2, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Rotated keypoints', sourceRotated)

# Creating Brute Force Matcher
bruteForce = cv2.BFMatcher()
matches = bruteForce.knnMatch(descriptors1, descriptors2, k = 2)

# visualize the matches
print 'All possible matches:', len(matches)
dist = [i[0].distance for i in matches]

print 'distance: min: %.3f' % min(dist)
print 'distance: max: %.3f' % max(dist)

# Threshold test
# You can change threshold number here
thresholdKeypoints = [i[0] for i in matches if i[0].distance < 45]

# Ratio test
ratioKeypoints = []
for i, j in matches:
# You can change ratio threshold number here
    if i.distance > 0.01 * j.distance:
        ratioKeypoints.append([i])

print 'Threshold keypoints:', len(thresholdKeypoints)
print 'Ratio keypoints:', len(ratioKeypoints)

# Creating images for showing matches
h1, w1 = gray.shape[:2]
h2, w2 = grayRotated.shape[:2]
destination1 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
destination1[:h1, :w1, 0] = gray
destination1[:h2, w1:, 0] = grayRotated
destination1[:, :, 1] = destination1[:, :, 0]
destination1[:, :, 2] = destination1[:, :, 0]

destination2 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
destination2[:h1, :w1, 0] = gray
destination2[:h2, w1:, 0] = grayRotated
destination2[:, :, 1] = destination2[:, :, 0]
destination2[:, :, 2] = destination2[:, :, 0]

ratioCounter, thresholdCounter = 0, 0

# Drawing matched keypoints and rotating test
for m in thresholdKeypoints:
    if abs(-int(keypoints1[m.queryIdx].pt[1]) + w2 - int(keypoints2[m.trainIdx].pt[0])) < 2 and abs(int(keypoints1[m.queryIdx].pt[0]) - int(keypoints2[m.trainIdx].pt[1])) < 2:
        color = tuple([np.random.randint(0, 255) for _ in xrange(3)])
        cv2.line(destination1, (int(keypoints1[m.queryIdx].pt[0]), int(keypoints1[m.queryIdx].pt[1])), (int(keypoints2[m.trainIdx].pt[0]) + w1, int(keypoints2[m.trainIdx].pt[1])), color)
    else:
        thresholdCounter += 1

for m in ratioKeypoints:
    if abs(-int(keypoints1[m[0].queryIdx].pt[1]) + w2 - int(keypoints2[m[0].trainIdx].pt[0])) < 2 and abs(int(keypoints1[m[0].queryIdx].pt[0]) - int(keypoints2[m[0].trainIdx].pt[1])) < 2:
        color = tuple([np.random.randint(0, 255) for _ in xrange(3)])
        cv2.line(destination2, (int(keypoints1[m[0].queryIdx].pt[0]), int(keypoints1[m[0].queryIdx].pt[1])), (int(keypoints2[m[0].trainIdx].pt[0]) + w1, int(keypoints2[m[0].trainIdx].pt[1])), color)
    else:
        ratioCounter += 1

#Showing results
cv2.imshow('Threshold matches', destination1)
cv2.imshow('Ratio matches', destination2)

print 'Threshold correct matches: %.3f' % ((len(thresholdKeypoints) - thresholdCounter) / (len(thresholdKeypoints) / 100.0)), '%'
print 'Ratio correct matches: %.3f' % ((len(ratioKeypoints) - ratioCounter) / (len(ratioKeypoints) / 100.0)), '%'

#waiting ESC key
while True:
    key = cv2.waitKey(1)
    if key == 27: break
