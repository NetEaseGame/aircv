aircv
=====

Python Lib based on python-opencv2 *for python2.7+*

### Usage

    import aircv as ac

SIFT查找图像

    imsrc = ac.imread('youimage.png')
    print ac.find_sift(imsrc)

期望如下输出, 查找失败就是None

    ((215, 45), [(160, 24), (161, 66), (270, 66), (269, 24)])
    
## DEVELOPING

LICENCE under [MIT](LICENSE)