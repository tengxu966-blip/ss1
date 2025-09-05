import cv2
import numpy as np

# 模拟DRACO图像：1024x1024灰度，1像素目标
img_size = 1024
img = np.zeros((img_size, img_size), dtype=np.uint8)
target_pos = (512, 512)  # 目标位置
img[target_pos] = 150  # 灰度值（弱信号）

# 添加噪声
noise = np.random.normal(0, 20, img.shape).astype(np.uint8)
img_noisy = cv2.add(img, noise)

# 预处理
img_preprocessed = cv2.medianBlur(img_noisy, 3)

# 阈值化
thresh_value = 100
_, thresh = cv2.threshold(img_preprocessed, thresh_value, 255, cv2.THRESH_BINARY)

# Blob检测与质心
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_intensity = 100  # 最小强度过滤
detected = None
for cnt in contours:
    M = cv2.moments(cnt)
    if M['m00'] != 0 and np.mean(img_noisy[cnt[:,0,1], cnt[:,0,0]]) > min_intensity:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        detected = (cx, cy)

# 融合轨道数据
expected_cx, expected_cy = 512, 512  # 从轨道预测
tolerance = 5  # 子像素容差
if detected and abs(detected[0] - expected_cx) < tolerance and abs(detected[1] - expected_cy) < tolerance:
    print(f"确认1像素目标: ({detected[0]}, {detected[1]})")
else:
    print("未确认目标")