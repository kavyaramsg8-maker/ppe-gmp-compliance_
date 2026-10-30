import cv2
import os
import time

print("Starting Live Monitor for 2 Cameras...")
print("Press 'q' in any window to quit.")

# Create blank black images for when the cameras aren't connected yet
blank = np.zeros((480, 640, 3), dtype=np.uint8) if 'np' in globals() else None
import numpy as np
blank = np.zeros((480, 640, 3), dtype=np.uint8)

while True:
    # 1. Check Camera 01
    if os.path.exists("live_CAM_01.jpg"):
        frame1 = cv2.imread("live_CAM_01.jpg")
        if frame1 is not None:
            cv2.imshow("CAM_01 (Corridor)", frame1)
            
    # 2. Check Camera 02
    if os.path.exists("live_CAM_02.jpg"):
        frame2 = cv2.imread("live_CAM_02.jpg")
        if frame2 is not None:
            cv2.imshow("CAM_02 (Production)", frame2)

    # Press 'q' to quit
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()