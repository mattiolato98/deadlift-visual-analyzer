# deadlift-visual-analyzer
# Why

We present YFGD (Your First Good Deadlift) for barbell deadlift technical evaluation. The purpose of our project is to offer a comprehensive evaluation and characterization of the execution of a specific physical exercise in the context of sports. Our computer vision initiative is intended to aid individuals who engage in solo gym training and seek to evaluate the accuracy of their exercise execution.

This is the project of the course of Computer Vision and Cognitive Systems of the University of Modena and Reggio Emilia.

# Installation
Clone the repo and install the requirements.

```bash
pip install -r requirements.txt
```

**Note**

The system has been tested in a Python 3.8.4 environment. The required packages versions may not work in another Python version. The overall system funcionality is not guaranteed in a different version.

# Inference

You could either run the inference interactively, choosing the options while running, or directly by command line, adding the appropriate options. The only mandatory parameter is the path to the video to evaluate.

## Interactive mode

```bash
python main.py <video_path>
```

## Command line mode

There are two groups of mutually exclusive parameters, regarding:

* Automatic barbell detection, `-y` or `--yolo-detection` to use YOLOv5 automatic detection, `-m` or `--mean-shift` to manual select the barbell and use mean shift tracking. Note that the second option is not suitable in CLI only systems, since it's necessary to select the barbell in the first frame of the video.
* Saving single repetitions to filesystem, `-s` or `--save-reps` to save,`-n` or `--no-save-reps` not to save.

**Note** that if you don't specify an option for each of the groups, the specific option will be then required in interactive mode during the execution.

### Examples

Use YOLOv5 detection and save the single repetitions to filesystem.

```bash
python main.py <video_path> -ys
```

Use YOLOv5 detection and **don't** save the single repetitions to filesystem.

```bash
python main.py <video_path> -yn
```

Use mean shift tracking and save the single repetitions to filesystem.

```bash
python main.py <video_path> -ms
```


Use mean shift tracking and **don't** save the single repetitions to filesystem.

```bash
python main.py <video_path> -mn
```

## Mean shift tracking 

If you choose mean shift as tracking algorithm, a window with the first frame will show up. Select the weight plates dragging the cursor from the top left corner to the bottom right one. For optimal results, it is recommended to choose the closer side of the barbell. Press the spacebar to start the tracking process once your selection is complete.

## Saving repetitions to filesystem

When you choose to save videos to filesystem (`-s`, `--save-reps` or through the interactive mode), the results will be available at the end of the execution at `<project_root_directory>/results/<video_name>/`.