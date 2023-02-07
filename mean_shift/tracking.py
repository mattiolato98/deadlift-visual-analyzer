import cv2
import numpy as np

from motion_detection.motion_utils import MotionDetector


def mean_shift_motion_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    # take first frame of the video
    ret, frame = cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_height, frame_width = frame.shape[:2]
    frame = cv2.resize(frame, [frame_width//2, frame_height//2])

    # setup initial location of window
    x, y, w, h = cv2.selectROI('Select barbell', frame, showCrosshair=True)
    cv2.destroyWindow('Select barbell')

    track_window = (x, y, w, h)
    # set up the ROI for tracking
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Set up the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    frame_number = 1
    motion_detector = MotionDetector(fps=fps, threshold=10, frame_number=frame_number)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, [frame_width // 2, frame_height // 2])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('img2', img2)

        # Detect if the current frame is a motion frame
        motion_detector.detect_motion(y, frame_number)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        frame_number += 1

    cv2.destroyAllWindows()

    return motion_detector.motion_frames
