"""
opencv version >= 3.0.0
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


img1 = cv2.imread('testdata/yl/q_big.png')
img2 = cv2.imread('testdata/yl/bg_half.png')


sift = cv2.ORB_create(edgeThreshold=20)


kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
print len(kp1), len(kp2)


bf = cv2.BFMatcher(cv2.NORM_HAMMING)#, crossCheck=True)
matches = bf.match(des1, des2)
#matches = bf.knnMatch(des1, des2, k=2)

print len(matches)

"""
good = []
for m, n in matches:
    print m.distance, n.distance
    if m.distance < 0.90 * n.distance:
        good.append([m])
matches = good

"""
matches = sorted(matches, key=lambda v: v.distance)
matches = matches[:100]
# for m in matches:
#     print m.distance

img3 = "output.png"
size = (height, width, channels) = (512, 256, 3)
img3 = np.zeros(size, np.uint8)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, img3, flags=2)
#img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img3, flags=2)
plt.imshow(img3), plt.show()
