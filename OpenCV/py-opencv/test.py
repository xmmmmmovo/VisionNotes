import numpy as np
import cv2

g = np.uint8([[[0, 255, 0]]])
r = np.uint8([[[0, 0, 255]]])

print(cv2.cvtColor(g, cv2.COLOR_RGB2HSV))
print(cv2.cvtColor(r, cv2.COLOR_RGB2HSV))