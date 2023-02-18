import cv2
import os

from paths import RESULTS_PATH as SAVE_PATH


class Video:
    """A class that represents a video file and provides methods to manipulate and extract
    information from it.
    """
    def __init__(self, video, motion_frames):
        """Initializes a new instance of the Video class.

        Args:
            video (str): The path to the video file.
            motion_frames (List[int]): A list of frame numbers with barbell in motion.
        """
        self.cap = cv2.VideoCapture(video)
        self.motion_frames = motion_frames

        self.n_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.name = video.split('/')[-1].split('.')[0]

    def save_reps(self, reps, preds):
        """Saves single repetitions to the file system.

        Args:
            reps (dict): A dictionary of repetition numbers and corresponding frames.
            preds (list): A list of predictions for each repetition.
        """
        print(f"Saving your labeled repetitions in '{SAVE_PATH}/{self.name}' folder...")

        try:
            os.makedirs(SAVE_PATH / self.name)  # Creating directory to store repetitions videos
        except FileExistsError:
            pass

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        for (rep, frames), pred in zip(reps.items(), preds):
            out = cv2.VideoWriter(
                f'{SAVE_PATH}/{self.name}/rep{rep + 1}.mp4', fourcc, self.fps, (self.width, self.height)
            )
            for frame_number in frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                ret, input_frame = self.cap.read()
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"Repetition number {rep + 1}: " + "Good" if pred else "Bad"
                position = (50, 50)
                font_scale = 1
                color = (255, 255, 255)  # colore del testo in RGB
                thickness = 2
                cv2.putText(input_frame, text, position, font, font_scale, color, thickness)

                if not ret:
                    break

                out.write(input_frame)
            out.release()

        print(f"Saving successfully completed")
