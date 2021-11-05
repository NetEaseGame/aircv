#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  hzsunshx
# Created: 2015-03-23 14:42

"""
sift
"""
import cv2
import numpy as np

import aircv as ac

def sift_test():
    t1 = ac.imread("testdata/1s.png")
    t2 = ac.imread("testdata/2s.png")
    print(ac.sift_count(t1), ac.sift_count(t2))

    result = ac.find_sift(t1, t2, min_match_count=ac.sift_count(t1)*0.4) # after many tests, 0.4 may be best
    if result:
        print('Same')
    else:
        print('Not same')

def tmpl_test(imgsrc, imgtgt):
    t1 = ac.imread(imgsrc)
    t2 = ac.imread(imgtgt)
    import time
    start = time.time()
    print(ac.find_all_template(t1, t2))
    print('Time used:', time.time() - start)


def add_alpha(imgsrc, imgtgt):
    # a tool to generate 4 channel image
    img = cv2.imread(imgsrc)

    b_channel, g_channel, r_channel = cv2.split(img)

    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    # 最小值为0
    alpha_channel[:, :int(b_channel.shape[1] / 2)] = 100

    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    cv2.imwrite(imgtgt, img_BGRA)


def to_grayscale(imgsrc, imgtgt):
    # a tool to generate grayscale image
    img = cv2.imread(imgsrc)

    channel = img.shape[2]
    if channel == 1:
        img_gray = imgsrc
    elif channel == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif channel == 4:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    cv2.imwrite(imgtgt, img_gray)


if __name__ == '__main__':
    # sift_test()

    # add_alpha('testdata/2s.png','testdata/2s_trans.png')
    # to_grayscale('testdata/2s.png','testdata/2s_gray.png')

    tmpl_test("testdata/2s.png", "testdata/2t.png")
    # test find template in 3 channel images
    tmpl_test("testdata/2s_trans.png", "testdata/2t.png")
    # test find template in 4 channel images
    tmpl_test("testdata/2s_gray.png", "testdata/2t.png")
    # test find template in 1 channel images

