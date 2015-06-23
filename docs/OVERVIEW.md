# aircv
Python Lib based on python-opencv2 *for python2.7+*

## Usage

    import aircv as ac
    imsrc = ac.imread('youimage.png') # 原始图像
    imsch = ac.imread('searched.png') # 带查找的部分

#### SIFT查找图像

    print ac.find_sift(imsrc, imsch)

期望如下输出, 查找失败就是[]

中点坐标， 目标图像周围四个点的坐标

    ((215, 45), [(160, 24), (161, 66), (270, 66), (269, 24)])

#### SIFT多个相同的部分查找

    print ac.find_all_sift(imsrc, imsch, maxcnt = 0)

maxcnt是可选参数，限制最多匹配的数量。输出eg， 一个都找不到返回None

    [((215, 45), [(160, 24), (161, 66), (270, 66), (269, 24)])]

#### 直接匹配查找图像

    print ac.find_template(imsrc, imsch)

期望输出 (目标图片的中心点，相似度)

    (294, 13), 0.9715

查找多个相同的图片，如在图形

![template1](testdata/2s.png)

中查找

![template2](testdata/2t.png)

    print ac.find_all_template(imsrc, imsch)

期望输出 (目标图片的中心点，相似度)

    [((294, 13), 0.9715), ...]

效果

![2res](testdata/2res.png)


## DEVELOPING

LICENCE under [MIT](LICENSE)
