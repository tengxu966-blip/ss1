import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建合成图像：黑色背景上白色圆形（模拟小行星）
img = np.zeros((500, 500), dtype=np.uint8)
cv2.circle(img, (250, 250), 50, 255, -1)  # 中心在 (250,250)，半径 100

# 添加噪声模拟太空相机效果
noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
img_noisy = cv2.add(img, noise)

# 预处理并检测圆形
gray = cv2.medianBlur(img_noisy, 5)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        print(f"检测到中心: ({x}, {y})，半径: {r}")

# 可视化
plt.imshow(img_noisy, cmap='gray')
plt.show()