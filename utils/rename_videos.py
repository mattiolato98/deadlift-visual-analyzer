import os
import subprocess
cl = ["Good", "Bad"]

for label in cl:
    directory = os.getcwd() + f"/deadlift_videos/{label}"
    i = 0
    for filename in os.listdir(directory):
        i += 1
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            command = f"mv {f} {directory}/{i}.mp4"
            subprocess.call(command, shell=True)
