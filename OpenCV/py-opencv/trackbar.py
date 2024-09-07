import cv2
import numpy as np


def n(x):
    pass


img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')

# 创建RGB三个滑动条
cv2.createTrackbar('R', 'image', 0, 255, n)
cv2.createTrackbar('G', 'image', 0, 255, n)
cv2.createTrackbar('B', 'image', 0, 255, n)

while (True):
    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        break

    # 获取滑块的值
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    # 设定img的颜色
    img[:] = [b, g, r]
