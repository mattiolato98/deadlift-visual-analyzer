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
        self._motion_frames = []

    @property
    def mean_y(self):
        return mean(self.positions) if len(self.positions) > 0 else 0

    @property
    def motion_frames(self):
        """Before returning motion frames list, extends it adding 1 * fps frame values before each gap.
        Since motion frames are ideally recorded when the barbell is moving, then it is necessary to
        go back of a small amount of frames to catch the start of a repetition.
        """
        sec = int(self.fps) * 1

        extended_motion_frames = copy.deepcopy(self._motion_frames)

        for idx, value in enumerate(extended_motion_frames):
            if value - extended_motion_frames[idx - 1] > sec:
                extended_motion_frames.extend([i for i in range(value - sec, value)])

        extended_motion_frames.sort()

        return extended_motion_frames

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
            self._motion_frames.append(frame)
