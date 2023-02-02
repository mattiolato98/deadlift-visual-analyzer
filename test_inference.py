import os
from collections import defaultdict

from main import inference

PATH = 'test/videos/'


if __name__ == '__main__':
    results = defaultdict(list)
    for idx, filename in enumerate(os.listdir('test/videos')):
        if os.path.isfile(f'{os.getcwd()}/{PATH}{filename}'):
            print(f'\n{idx + 1}) Processing {filename}')

            predictions = inference(f'{PATH}{filename}')
            results[filename].append(predictions)

            print(f'\n{idx + 1}) {filename} DONE.\n\n')
            print(results)

    print(results)