import argparse
import os
import sys

from roboflow import Roboflow

IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 100
WEIGHTS_PATH = 'custom_weights/yolov5s.pt'


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
        f'--data {dataset_location}/data.yaml --weights {WEIGHTS_PATH} --name yolov5s_results  --cache'
    )


def val(dataset_location):
    os.system(
        f'python yolov5/val.py --weights custom_weights/best.pt '
        f'--data {dataset_location}/data.yaml --task val'
    )


if __name__ == '__main__':
    sys.path.insert(0, "yolov5")

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', action='store_true', help='Load dataset from Roboflow before training')
    parser.add_argument('-v', '--val', action='store_true', help='Validation')
    args = parser.parse_args()

    if args.load:
        dataset_location = load_dataset()
    else:
        print('\nSkipping dataset load ...\n')
        with open('dataset_location.txt', 'r') as f:
            dataset_location = f.read().strip()

    if args.val:
        val(dataset_location)
    else:
        train(dataset_location)
