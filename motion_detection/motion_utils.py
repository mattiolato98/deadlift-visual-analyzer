import copy

from statistics import mean


class MotionDetector:
    def __init__(self, fps=30, threshold=10, frame_number=0):
        self.fps = fps
        self.barbell_motion = []
        self.positions = []
        self.first_motion = False
        self.threshold = threshold
        self.frame_number = frame_number
        self.motion_frames = []

    @property
    def mean_y(self):
        return mean(self.positions) if len(self.positions) > 0 else 0

    def build_ground_position(self, y):
        """ Compute the average y value, when the barbell is on the ground.
        Args:
            y: current position of the bounding box along y-axis
        """
        self.positions.append(y)

    def is_motion_frame(self, y):
        """ Detect if the current frame is a motion frame (a frame in which the barbell is not on the ground).
        Args:
            y: current position of the bounding box along y-axis

        Returns:
            True if bb movement along y-axis exceeds threshold
        """
        if abs(self.mean_y - y) > self.threshold:
            return True

        return False

    def detect_motion(self, y, frame):
        if not self.first_motion:
            self.build_ground_position(y)
            self.first_motion = self.is_motion_frame(y)

        if self.is_motion_frame(y):
            self.motion_frames.append(frame)
