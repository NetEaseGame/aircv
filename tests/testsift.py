# encoding=utf-8
"""
OpenCV version = 2.x
"""

import numpy as np
import cv2
import random
from matplotlib import pyplot as plt


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1, :] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2, cols1:cols1 + cols2, :] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        mat = mat

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour random
        f = lambda : random.randint(1, 255)
        line_color = (f(), f(), f())
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), line_color, 1)

    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_show(src, dst):
    img1 = cv2.imread(dst, 0)
    img2 = cv2.imread(src, 0)

    # the larger edgeThreshold is, the more sift keypoints we find 
    sift = cv2.SIFT(edgeThreshold=100)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    print len(matches), "sift feature points found"

    good = []
    for m, n in matches:
        # print m.distance, n.distance, m.distance / n.distance
        # filter those pts similar to the next good ones
        if m.distance < 0.9 * n.distance:
            good.append(m)
    print len(good), "good feature points"

    # require count >= 4 in function cvFindHomography
    if len(good) >= 4:
        sch_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        img_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M是转化矩阵
        M, mask = cv2.findHomography(sch_pts, img_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # 计算四个角矩阵变换后的坐标，也就是在大图中的坐标
        h, w = img1.shape[:2]
        pts = np.float32(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # trans numpy array to python list
        # [(a, b), (a1, b1), ...]
        pypts = []
        for npt in dst.astype(int).tolist():
            pypts.append(tuple(npt[0]))

        lt, br = pypts[0], pypts[2]
        middle_point = (lt[0] + br[0]) / 2, (lt[1] + br[1]) / 2

        result = dict(
            result=middle_point,
            rectangle=pypts,
            confidence=min(1.0 * matchesMask.count(1) / 10, 1.0)
        )
        print result

        selected = []
        for k, v in enumerate(good):
            if matchesMask[k]:
                selected.append(v)
        print len(selected), "selected by homography"

    else:
        raise Exception("not enough matches found %s/%s" % (len(good), 4))

    # all sift feature points
    drawMatches(img1, kp1, img2, kp2, [i[0] for i in matches])
    # good feature points, good one distance/next good one distance <= 90%
    drawMatches(img1, kp1, img2, kp2, good)
    # select points by homography, those of same matrix transformation 
    drawMatches(img1, kp1, img2, kp2, selected)


if __name__ == '__main__':
    find_show("testdata/g18/screen_big2.png", "testdata/g18/task2.png")
    # find_show("testdata/2s.png", "testdata/2t.png")
    # find_show("testdata/yl/bg_2.5.png", "testdata/yl/q_small.png")
    # find_show("testdata/yl/bg_2.png", "testdata/yl/q_big.png")
    # find_show("testdata/xyq/screen.png", "testdata/xyq/bk.png")
    # find_show("testdata/xyq/screen.png", "testdata/xyq/bp.png")
    # find_show("testdata/xyq/screen.png", "testdata/xyq/fb.png")
    # find_show("testdata/xyq/screen.png", "testdata/xyq/sd.png")
    # find_show("testdata/xyq/screen.png", "testdata/xyq/xz.png")
