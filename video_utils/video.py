from pathlib import Path

import cv2
import os


class Video:
    def __init__(self, video, motion_frames):
        self.cap = cv2.VideoCapture(video)
        self.motion_frames = motion_frames

        self.n_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.name = video.split('/')[-1].split('.')[0]

    def save_reps(self, save_path, reps, preds):
        """ Save single repetitions to filesystem. """
        try:
            os.mkdir(save_path)  # Creating directory to store repetitions videos
        except FileExistsError:
            pass

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        for (rep, frames), pred in zip(reps.items(), preds):
            out = cv2.VideoWriter(
                f'{save_path}/rep{rep + 1}.mp4', fourcc, self.fps, (self.width, self.height)
            )
            for frame_number in frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                ret, input_frame = self.cap.read()
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"Repetition n{rep}: " + "Good" if pred else "Bad"
                position = (50, 50)
                font_scale = 1
                color = (255, 255, 255)  # colore del testo in RGB
                thickness = 2
                cv2.putText(input_frame, text, position, font, font_scale, color, thickness)

                if not ret:
                    break

                out.write(input_frame)
            out.release()
