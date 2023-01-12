import argparse
import os

import cv2
import numpy as np
import pandas as pd
import tqdm

from custom_yolov5.detect_motion import detect_motion_frames
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

from pose_classification.ema_smoothing import EMADictSmoothing
from pose_classification.pose_classifier import PoseClassifier
from pose_classification.pose_embedder import FullBodyPoseEmbedder
from rep_counter.repetition_counter import RepetitionCounter

WEIGHTS_PATH = 'custom_yolov5/best.pt'
PATH = 'test/videos/'
CLASS_NAME = 'deadlift_up'
POSE_SAMPLES_PATH = 'deadlift_poses'
GROUND_TRUTH_CSV = 'test/reps.csv'


def count_repetitions(
        cap,
        video_n_frames,
        pose_tracker,
        pose_classifier,
        pose_classification_filter,
        repetition_counter):
    old_reps = 0
    with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
        while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) != motion_frames[-1]:
            # Get next frame of the video.
            ret, input_frame = cap.read()
            if not ret:
                break
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
                    old_reps = repetitions_count
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

    print(f'\n\nFINAL REPS: {repetition_counter.n_repeats}\n')

    return repetition_counter.n_repeats


def process_video(video, motion_frames):
    cap = cv2.VideoCapture(video)

    # Get some video parameters to generate output video with classification
    video_n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if motion_frames is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, motion_frames[0])

    pose_tracker = mp_pose.Pose(upper_body_only=False)
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

    total_repetitions = count_repetitions(
        cap, video_n_frames, pose_tracker, pose_classifier, pose_classification_filter, repetition_counter
    )

    return total_repetitions


def check_errors(df):
    ground_truth_df = pd.read_csv(GROUND_TRUTH_CSV)

    result = pd.concat([df, ground_truth_df]).groupby('video').sum()
    result['diff'] = result['repetitions'] - result['computed_repetitions']

    return result[result['diff'] != 0]


def test():
    df = pd.DataFrame(columns=['video', 'computed_repetitions'])

    total_videos = sum(1 for video in os.listdir(PATH) if os.path.isfile(f'{os.getcwd()}/{PATH}{video}'))
    current = 1

    print(f'\nRunning test on {total_videos} videos...\n')

    for video in os.listdir(PATH):
        video_path = f'{os.getcwd()}/{PATH}{video}'
        if os.path.isfile(video_path):
            print(f'\n\U0001f3cb Starting processing {current}/{total_videos}: {video}\n\n')
            motion_frames = detect_motion_frames(
                max_det=1,
                weights=WEIGHTS_PATH,
                conf_thres=0.4,
                source=video_path,
            )
            print(motion_frames)

            repetitions = process_video(
                video_path,
                motion_frames if len(motion_frames) > 0 else None
            )

            df = pd.concat([
                df, pd.DataFrame([[video, repetitions]], columns=['video', 'computed_repetitions'])
            ])

            print(f'Video {current}/{total_videos} done.\n')
            print(df)

            print('\n\n-------------------------------------------------------------------------------------------\n\n')

            current += 1

    errors_df = check_errors(df)

    df.to_csv('computed_reps.csv')
    errors_df.to_csv('errors.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--test',
        action='store_true',
        help='Run over a set of selected videos, and compare the result with the ground truth.'
    )
    args = parser.parse_args()

    print('\n\n\U0001f680 Repetition counter by Gabriele Mattioli, Sara Morandi, Filippo Rinaldi\n')

    if args.test:
        test()
    else:
        video_path = input("Enter the path of the video to classify: ")
        if not os.path.isfile(video_path):
            raise FileNotFoundError()

        motion_frames = detect_motion_frames(
            max_det=1,
            weights=WEIGHTS_PATH,
            conf_thres=0.4,
            source=video_path,
        )
        print(motion_frames)

        repetitions = process_video(
            video_path,
            motion_frames if len(motion_frames) > 0 else None
        )

        print(f'\nTotal video repetitions: {repetitions}\n\n')
