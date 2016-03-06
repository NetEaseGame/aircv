#!/usr/bin/env python
# coding: utf-8
#
# aircv test cases

import aircv

def main():
    bg = aircv.imread('testdata/bg.png')
    search = aircv.imread('testdata/guard.png')
    
    print aircv.find_sift(bg, search)

if __name__ == '__main__':
    main()