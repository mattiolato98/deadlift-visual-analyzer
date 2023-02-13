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
            print(predictions)
            results[filename] = predictions
            for id, pred in enumerate(results[filename]):
                # print(pred)
                print(f"Repetition number {id + 1} of {len(predictions)} : judged ", "Good" if pred else "Bad")

    for video in results:
        show_results(video, results[video])