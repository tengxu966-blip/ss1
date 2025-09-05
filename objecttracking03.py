import cv2

# 目前OpenCV4.2版本中自带了8个目标跟踪算法的实现。
#
# BOOSTING：算法原理类似于Haan cascades（AdaBoost），是一种很老的算法。这个算法速度慢并且不是很准。
# MIL：把BOOSTING推一点。
# KCF：速度比BOOSTING和MIL更快，与BOOSTING和MIL一样不能很好地处理递归问题。
# CSRT：比较更难一些，但是速度比KCF稍慢。
# MedianFlow：对于快速移动的目标和外形变化迅速的目标效果不好。
# TLD：会产生较多的false-positives。
# MOSSE：算法速度非常快，但是准确率比不上KCF和CSRT。在一些运算算法速度的场合很适用。
# GOTURN：OpenCV中自带的唯一一个基于深度学习的算法。运行算法需要提前下载并模拟文件。
# 综合算法速度和准确率考虑，个人觉得CSRT、KCF、MOSSE这三个目标跟踪算法较好。
#     "csrt": cv2.legacy.TrackerCSRT_create,
#     "kcf": cv2.legacy.TrackerKCF_create,
#     "boosting": cv2.legacy.TrackerBoosting_create,
#     "mil": cv2.legacy.TrackerMIL_create,
#     "tld": cv2.legacy.TrackerTLD_create,
#     "medianflow": cv2.legacy.TrackerMedianFlow_create,
#     "mosse": cv2.legacy.TrackerMOSSE_create

tracker = cv2.TrackerCSRT_create()

cap = cv2.VideoCapture("videos/traffic.mp4")

ret, frame = cap.read()

bbox = cv2.selectROI('Frame', frame, fromCenter=False, showCrosshair=True)

tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker
    ok, box = tracker.update(frame)

    if ok:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("frame", frame)

    if cv2.waitKey(50) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()