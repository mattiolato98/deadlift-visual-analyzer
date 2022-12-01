import os
import subprocess
cl = ["Bad"]

new_dataset = os.getcwd() + f"/deadlift_downscaled_360p/"
width = 480
height = 360
if not os.path.exists(new_dataset):
    os.mkdir(new_dataset)
for label in cl:
    directory = os.getcwd() + f"/test/{label}"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            # os.chdir(new_dataset)
            new_dir = os.path.join(new_dataset, label)
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            command = f"ffmpeg -i {f} -vf scale={width}:{height} {new_dir}/{filename}"
            subprocess.call(command, shell=True)
