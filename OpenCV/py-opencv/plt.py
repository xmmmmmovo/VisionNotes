import cv2
import matplotlib.pyplot as plt

img = cv2.imread('jp.png')
# img2 = img[:, :, ::-1]
# 或使用
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 显示不正确的图
plt.subplot(121), plt.imshow(img)
plt.xticks([]), plt.yticks([])  # 隐藏x和y轴

# 显示正确的图
plt.subplot(122)
plt.xticks([]), plt.yticks([])  # 隐藏x和y轴
plt.imshow(img2)

plt.show()
