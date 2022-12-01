import os
import subprocess
cl = ["Good", "Bad"]

dataset = os.getcwd() + f"/Dataset_downscaled/openpose/"
for label in cl:
    directory = dataset+cl
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        file = os.path.splitext(filename)[0]
        if os.path.isfile(f):
            command = f"ffmpeg -i {f} -c:v copy -c:a copy -y {file}.mp4"
            subprocess.call(command, shell=True)
