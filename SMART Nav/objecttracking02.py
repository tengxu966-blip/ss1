import cv2
# opencv中常用的背景分割器有三种：MOG2、KNN、GMG

def filter_img(frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Fill any small holes
    closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # Dilate to merge adjacent blobs
    dilated = cv2.dilate(opening, kernel, iterations=2)

    # _, thresh = cv2.threshold(frame, 244, 255, cv2.THRESH_BINARY)
    # thresh = cv2.erode(thresh, kernel, iterations=2)
    # dilated = cv2.dilate(thresh, kernel, iterations=2)

    return dilated

# Object detection from Stable camera
# Object detection from Stable camera
# MOG2Z算法阴影检测并不充美，但它有助于将目标轮廓按原始形状进行还原;

# KNN的精确性和阴影检测能力较好，即使是相邻对象也没有在一起检测，运动检测的结果精确。

# object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)
# object_detector = cv2.createBackgroundSubtractorKNN(detectShadows = True)
# object_detector = cv2.bgsegm.createBackgroundSubtractorGMG()
    cap = cv2.VideoCapture("videos/traffic.mp4")

    while True:
        ret, frame = cap.read()

    # 1. Object Detection
        mask = object_detector.apply(frame)
        mask = filter_img(mask)

        # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        # (x, y, w, h) = cv2.boundingRect(contour)
        # if cv2.contourArea(contour) < 500:
        # continue
        # cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
        # cv2.drawContours(frame, contours, -1, (0,255, 0), 2)

        # show the output frame
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        if cv2.waitKey(50) == ord("q"):
            break

        cap.release()
        cv2.destroyAllWindows()