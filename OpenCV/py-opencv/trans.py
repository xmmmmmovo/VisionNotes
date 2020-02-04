import cv2

img = cv2.imread('drawing.jpg')

# 按照指定的宽度、高度缩放图片
res = cv2.resize(img, (132, 150))
# 按照比例缩放，如x,y轴均放大一倍
res2 = cv2.resize(
    img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR
)

cv2.imshow('shrink', res), cv2.imshow('zoom', res2)

# 参数2 = 0：垂直翻转(沿x轴)
# 参数2 > 0: 水平翻转(沿y轴)
# 参数2 < 0: 水平垂直翻转
f = cv2.imread('lena.jpg')
cv2.imshow('l', cv2.flip(f, 1))

# 平移图片
import numpy as np

rows, cols = img.shape[:2]

# 定义平移矩阵，需要是numpy的float32类型
# x轴平移100，y轴平移50
M = np.float32([[1, 0, 100], [0, 1, 50]])
# 用仿射变换实现平移
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('shift', dst)

# 45°旋转图片并缩小一半
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.5)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('rotation', dst)
cv2.waitKey(0)