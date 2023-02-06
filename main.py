import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tqdm

from custom_yolov5.detect_motion import detect_motion_frames
from mean_shift.tracking import mean_shift_motion_frames
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

from pose_classification.ema_smoothing import EMADictSmoothing
from pose_classification.pose_classifier import PoseClassifier
from pose_classification.pose_embedder import FullBodyPoseEmbedder
from rep_counter.repetition_counter import RepetitionCounter
from slowfast.slowfast import inference as slowfast_inference

WEIGHTS_PATH = 'custom_weights/best.pt'
PATH = 'test/videos/'
CLASS_NAME = 'deadlift_up'
POSE_SAMPLES_PATH = 'deadlift_poses'
GROUND_TRUTH_CSV = 'test/reps.csv'


def count_and_split_repetitions(
        cap,
        video_n_frames,
        video_fps,
        video_width,
        video_height,
        motion_frames,
        pose_tracker,
        pose_classifier,
        pose_classification_filter,
        repetition_counter):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # out = cv2.VideoWriter('rep0.mp4', fourcc, video_fps, (video_width, video_height))
    reps = defaultdict(list)

    old_reps = 0
    start_frame_idx = 0
    with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
        for idx, frame_number in enumerate(motion_frames):
            # Get next frame of the video.
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, input_frame = cap.read()

            if not ret:
                break

            # out.write(input_frame)

            # Run pose tracker.
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks

            # Draw pose prediction.
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)

            if pose_landmarks is not None:
                # Get landmarks.
                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                           for lmk in pose_landmarks.landmark], dtype=np.float32)
                assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                # Classify the pose on the current frame.
                pose_classification = pose_classifier(pose_landmarks)

                # Smooth classification using EMA.
                pose_classification_filtered = pose_classification_filter(pose_classification)

                # Count repetitions.
                repetitions_count = repetition_counter(
                    pose_classification_filtered,
                    cap.get(cv2.CAP_PROP_POS_FRAMES))

                if repetitions_count > old_reps:
                    print(f'-\n--------------- Reps: {repetitions_count} ---------------\n')
                    reps[repetitions_count - 1].extend(motion_frames[start_frame_idx:idx+1])
                    start_frame_idx = idx + 1

                    old_reps = repetitions_count
                    # out.release()
                    # out = cv2.VideoWriter(f'rep{repetitions_count}.mp4', fourcc, video_fps, (video_width, video_height))
            else:
                # No pose => no classification on current frame.
                pose_classification = None

                # Still add empty classification to the filter to maintaining correct
                # smoothing for future frames.
                pose_classification_filtered = pose_classification_filter(dict())
                pose_classification_filtered = None

                # Don't update the counter presuming that person is 'frozen'. Just
                # take the latest repetitions count.
                repetitions_count = repetition_counter.n_repeats

                pbar.update()

    # Release MediaPipe resources.
    pose_tracker.close()
    # out.release()

    # Remove gaps in a repetition list of frames. A single repetition must not contain gaps.
    reps = remove_gaps(reps, video_fps)

    print(f'\n\nTotal video repetitions: {repetition_counter.n_repeats}\n')

    return reps, repetition_counter.n_repeats


def remove_gaps(rep_dict, fps):
    """Removes gaps in frames in a rep.
    A gap is defined as more than 1 * fps distance between two list elements.

    How it works:
        At first, split the list containing a single rep frames in more lists without gaps, for each rep.
        Then, it keeps the longest list for each rep.

    e.g. frames of second rep = [100, 101, 102, 133, 134, 135, 136] with fps = 30
         cleaned rep list becomes = [133, 134, 135, 136]
    """
    max_gap = fps
    cleaned_reps = defaultdict(list)

    for rep, frames in rep_dict.items():
        count = 0
        results = defaultdict(list)
        for idx, elem in enumerate(frames):
            if idx != 0 and elem - frames[idx - 1] > max_gap:
                count += 1
            results[count].append(elem)
        cleaned_reps[rep].extend(max(results.values(), key=len))

    return cleaned_reps


def process_video(video, motion_frames):
    cap = cv2.VideoCapture(video)

    # Get some video parameters to generate output video with classification
    video_n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_tracker = mp_pose.Pose()
    pose_embedder = FullBodyPoseEmbedder()
    pose_classifier = PoseClassifier(
        pose_samples_folder=POSE_SAMPLES_PATH,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10
    )
    pose_classification_filter = EMADictSmoothing(
        window_size=int(video_fps),
        alpha=0.2
    )
    repetition_counter = RepetitionCounter(
        class_name=CLASS_NAME,
        motion_frames=motion_frames,
        enter_threshold=0.90,
        exit_threshold=0.49
    )

    reps_frames, total_repetitions = count_and_split_repetitions(
        cap, video_n_frames, video_fps, video_width, video_height, motion_frames, pose_tracker, pose_classifier, pose_classification_filter, repetition_counter
    ) if motion_frames is not None else 0

    return video_fps, reps_frames, total_repetitions


def inference(video_path, yolo_detection):
    if not os.path.isfile(video_path):
        raise FileNotFoundError()

    if yolo_detection:
        motion_frames = detect_motion_frames(
            max_det=1,
            weights=WEIGHTS_PATH,
            conf_thres=0.4,
            source=video_path,
        )
        if len(motion_frames) == 0:
            manual = input("No barbell detected. Do you want to try with manual tracking (y/n)? ").lower()
            if manual != 'n':
                inference(video_path, False)
            else:
                return None
    else:
        motion_frames = mean_shift_motion_frames(video_path)

    fps, reps_frames, total_repetitions = process_video(
        video_path,
        motion_frames if len(motion_frames) > 0 else None
    )

    reps_range = [(frames[0] / fps, frames[-1] / fps) for frames in reps_frames.values()]

    preds = slowfast_inference(video_path, reps_range) if len(reps_range) > 0 else 'No motion frames'

    return preds


def show_results(filename, predictions):
    if predictions is None:
        print('Nothing predicted')
    else:
        print(f'Processing of {filename} completed')
        print(f"Total repetitions : {len(predictions)}")
        for idx, pred in enumerate(predictions):
            print(f"Repetition number {idx + 1} of {len(predictions)} : judged ", "Good" if pred else "Bad")


if __name__ == '__main__':
    video_path = input("Enter the path of the video to classify: ")
    automatic_detection = input("Do you want to use automatic detection (y/n)? ").lower()
    if automatic_detection != 'n':
        automatic_detection = True
    else:
        automatic_detection = False

    preds = inference(video_path, automatic_detection)
    show_results(video_path, preds)
