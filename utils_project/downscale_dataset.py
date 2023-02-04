import os
import subprocess
from pathlib import Path
cl = ["Good", "Bad"]
path = Path(os.getcwd())
print(path.parent.absolute())
new_dataset = path.parent.absolute() / "Dataset_downscaled_540p"
width = 960
height = 540
if not os.path.exists(new_dataset):
    os.mkdir(new_dataset)
for label in cl:
    directory = path.parent.absolute() / f"Dataset/{label}"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            os.chdir(new_dataset)
            new_dir = os.path.join(new_dataset, label)
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            command = f"ffmpeg -i {f} -vf scale={width}:{height} {new_dir}/{filename}"
            subprocess.call(command, shell=True)
