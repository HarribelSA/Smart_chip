import cv2

class MotionDetector:
    def __init__(self, threshold=5000):
        self.prev_gray = None
        self.threshold = threshold

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_pixels = cv2.countNonZero(thresh)

        self.prev_gray = gray
        return motion_pixels > self.threshold