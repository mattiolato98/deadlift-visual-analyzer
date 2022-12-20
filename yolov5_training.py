import argparse
import os
import torch
import yaml

from roboflow import Roboflow
from yolov5.utils.downloads import attempt_download

IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 100


def load_dataset():
    with open('rf_api_key.txt', 'r') as f:
        rf = Roboflow(api_key=f.read().strip())

    project = rf.workspace('computer-vision-1bdzc').project('barbell-detection-yolov5')
    dataset = project.version(4).download('yolov5')

    os.system(f'cat {dataset.location}/data.yaml')

    with open('dataset_location.txt', 'w') as f:
        f.write(dataset.location)

    return dataset.location


def train(dataset_location):
    os.system(
        f'python yolov5/train.py --img {IMG_SIZE} --batch {BATCH_SIZE} --epochs {EPOCHS} '
        f'--data {dataset_location}/data.yaml --weights yolov5s.pt --name yolov5s_results  --cache'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', action='store_true', help='Load dataset from Roboflow before training')
    args = parser.parse_args()

    if args.load:
        dataset_location = load_dataset()
    else:
        print('\nSkipping dataset load ...\n')
        with open('dataset_location.txt', 'r') as f:
            dataset_location = f.read().strip()

    train(dataset_location)