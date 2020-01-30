import cv2
import numpy as np

img = cv2.imread('jp.png')

b = img[:, :, 2]
cv2.imshow('blue', b)

r = img[20:120, 50:220][:, :, 2]
cv2.imshow('red', r)
cv2.waitKey(0)