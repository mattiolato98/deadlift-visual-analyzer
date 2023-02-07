import os
from collections import defaultdict

from main import inference, show_results

PATH = 'test/videos/'


if __name__ == '__main__':
    # results = defaultdict(list)
    results = {}
    for idx, filename in enumerate(os.listdir('test/videos')):
        if os.path.isfile(f'{os.getcwd()}/{PATH}{filename}'):
            print(f'\n{idx + 1}) Processing {filename}')

            predictions = inference(f'{PATH}{filename}', yolo_detection=True)
            # results[filename].append(predictions)
            results[filename] = predictions

    for video in results:
        show_results(video, results[video])