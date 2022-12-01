import os
import subprocess
cl = ["1"]

new_dataset = os.getcwd() + f"/Stop_dataset_downscaled_540p/"
width = 960
height = 540
for label in cl:
    directory = os.getcwd() + f"/Stop_dataset/{label}"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            # os.chdir(new_dataset)
            new_dir = os.path.join(new_dataset, label)
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            command = f"ffmpeg -i {f} -vf scale={width}:{height} {new_dir}/{filename}"
            subprocess.call(command, shell=True)
