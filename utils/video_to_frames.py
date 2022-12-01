import os
import subprocess

dir_class = ["Bad", "Good"]  # Bad == 0, Good == 1
dataset = "/Dataset_downscaled_frames"
annotations_records = []

for label, cl in enumerate(dir_class):
    directory = os.getcwd() + f"{dataset}/{cl}"
    if not os.path.exists(directory):
        os.mkdir(directory)
    annotations = open(os.getcwd() + f"{dataset}/annotations.txt", "a")

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        file = os.path.splitext(filename)[0]
        if os.path.isfile(f):
            new_dir = os.path.join(directory, file)
            if not os.path.exists(new_dir):
                start_frame = 1
                os.mkdir(new_dir)
                command = f"ffmpeg -i {f} -r 30/1 {new_dir}/img_%05d.jpg"
                subprocess.call(command, shell=True)
                grep_cmd = f"ffmpeg -i {f} -vcodec copy -acodec copy -f null /dev/null 2>&1 | ggrep -oP '(?<=frame=  )\d+'"
                try:
                    end_frame = int(subprocess.check_output(grep_cmd, shell=True))
                except Exception:
                    grep_cmd = f"ffmpeg -i {f} -vcodec copy -acodec copy -f null /dev/null 2>&1 | ggrep -oP '(?<=frame=   )\d+'"
                    end_frame = int(subprocess.check_output(grep_cmd, shell=True))
                annotations_records.append(f"{cl}/{file} {start_frame} {end_frame} {label}\n")

annotations.writelines(annotations_records)
