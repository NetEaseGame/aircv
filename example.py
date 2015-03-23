#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  hzsunshx
# Created: 2015-03-23 14:42

"""
sift
"""

import aircv as ac

def sift_test():
    t1 = ac.imread("testdata/1s.png")
    t2 = ac.imread("testdata/1s-rotate.png")
    print ac.sift_count(t1), ac.sift_count(t2)

    print ac.find_sift(t1, t2, min_match_count=ac.sift_count(t1)*0.4) # after many tests, 0.4 may be best

if __name__ == '__main__':
    sift_test()

