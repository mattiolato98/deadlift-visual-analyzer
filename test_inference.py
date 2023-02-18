import os
from collections import defaultdict

from main import evaluation, show_results

PATH = 'test/video_bad/'


if __name__ == '__main__':
    """Process video files and show results."""
    results = {}
    for idx, filename in enumerate(os.listdir('test/video_bad')):
        if os.path.isfile(f'{os.getcwd()}/{PATH}{filename}'):
            print(f'\n{idx + 1}) Processing {filename}')

            predictions = evaluation(f'{PATH}{filename}', yolo_detection=False)
            # results[filename].append(predictions)
            print(predictions)
            results[filename] = predictions
            for id, pred in enumerate(results[filename]):
                # print(pred)
                print(f"Repetition number {id + 1} of {len(predictions)} : judged ", "Good" if pred else "Bad")

    for video in results:
        show_results(video, results[video])