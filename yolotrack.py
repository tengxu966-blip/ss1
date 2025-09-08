from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")

VIDEO_PATH = "videos/traffic.mp4"
RESULT_PATH = "result1.mp4"

# 记录所有id的位置信息
track_history = defaultdict(lambda: [])

if __name__ == '__main__':
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file.")
        exit()

    fps = capture.get(cv2.CAP_PROP_FPS)  # 获取帧率
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # 获取宽度
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获取高度

    print(f"原始视频帧率: {fps}")

    # 使用更高效的编码器
    fourcc = cv2.VideoWriter_fourcc(*"H264")  # 或者使用 "XVID", "MJPG"
    videoWriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (int(frame_width), int(frame_height)))

    # 设置窗口
    cv2.namedWindow("yolo track", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("yolo track", 640, 480)

    frame_count = 0
    start_time = time.time()

    while True:
        success, frame = capture.read()
        if not success:
            print("视频读取完成")
            break

        frame_count += 1

        # 可选：降低处理分辨率以提高速度
        # frame = cv2.resize(frame, (640, 480))

        results = model.track(frame, persist=True, verbose=False)  # verbose=False减少输出

        if results[0].boxes is not None and results[0].boxes.id is not None:
            # 可视化显示目标检测框
            a_frame = results[0].plot()

            # 所有id的位置信息
            boxes = results[0].boxes.xywh.cpu()
            # 所有ID的序列号信息
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:  # 减少轨迹点数量
                    track.pop(0)

                if len(track) > 1:
                    points = np.array(track).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(a_frame, [points], isClosed=False, color=(255, 0, 255), thickness=2)
        else:
            a_frame = frame.copy()

        # 写入视频帧
        videoWriter.write(a_frame)

        # 显示处理后的帧（可选，显示会降低速度）
        cv2.imshow("yolo track", a_frame)

        # 计算实际处理速度
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"处理速度: {frame_count / elapsed:.2f} FPS")

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 计算总处理时间
    total_time = time.time() - start_time
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"总帧数: {frame_count}")
    print(f"平均处理速度: {frame_count / total_time:.2f} FPS")

    capture.release()
    videoWriter.release()
    cv2.destroyAllWindows()
