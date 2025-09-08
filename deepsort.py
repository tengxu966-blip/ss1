from collections import defaultdict
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

import time

# 初始化YOLO模型（仅用于检测）
model = YOLO("yolov8n.pt")

# 初始化DeepSORT跟踪器
deepsort_tracker = DeepSort(
    max_age=50,  # 跟踪丢失的最大帧数
    n_init=3,  # 需要多少帧检测才能确认一个track
    max_iou_distance=0.7,
    max_cosine_distance=0.3,
    nms_max_overlap=1.0
)

VIDEO_PATH = "videos/traffic.mp4"
RESULT_PATH = "result_deepsort.mp4"

# 记录所有id的位置信息
track_history = defaultdict(lambda: [])

if __name__ == '__main__':
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file.")
        exit()

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (int(frame_width), int(frame_height)))

    cv2.namedWindow("DeepSORT Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DeepSORT Tracking", 640, 480)

    frame_count = 0
    start_time = time.time()

    while True:
        success, frame = capture.read()
        if not success:
            print("视频读取完成")
            break

        frame_count += 1

        # YOLOv8仅做目标检测（不使用track功能）
        results = model(frame, verbose=False)

        # 准备DeepSORT的检测输入
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()  # [x_center, y_center, width, height]
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                conf = confidences[i]
                class_id = class_ids[i]

                # 只检测人（class_id=0）或其他您需要的类别
                if class_id == 0 and conf > 0.5:  # 置信度阈值
                    detections.append(([x, y, w, h], conf, class_id))

        # DeepSORT进行跟踪
        tracks = deepsort_tracker.update_tracks(detections, frame=frame)

        # 创建显示帧
        display_frame = frame.copy()

        # 绘制DeepSORT的跟踪结果
        for track in tracks:
            if not track.is_confirmed():  # 只处理已确认的跟踪
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()  # [x1, y1, x2, y2]

            # 计算中心点（用于轨迹）
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            # 维护轨迹历史
            track_points = track_history[track_id]
            track_points.append((center_x, center_y))
            if len(track_points) > 30:
                track_points.pop(0)


            # 绘制轨迹
            if len(track_points) > 1:
                points = np.array(track_points).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(display_frame, [points], isClosed=False, color=(255, 0, 255), thickness=2)

            # # 绘制边界框和ID
            # cv2.rectangle(display_frame, (int(bbox[0]), int(bbox[1])),
            #               (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            # cv2.putText(display_frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 写入视频帧
        videoWriter.write(display_frame)

        # 显示处理后的帧
        cv2.imshow("DeepSORT Tracking", display_frame)

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