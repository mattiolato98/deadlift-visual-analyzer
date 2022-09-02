from video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os



videos_root = os.path.join(os.getcwd(), 'dataset/video_dataset_v2')
annotation_file = os.path.join(videos_root, 'annotations.txt')

preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(256),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(256),  # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

dataset = VideoFrameDataset(
    root_path=videos_root,
    annotationfile_path=annotation_file,
    num_segments=3,
    frames_per_segment=32,
    imagefile_template='img_{:05d}.jpg',
    transform=preprocess,
    test_mode=False
  )

sample = dataset[2]
frame_tensor = sample[0]  # tensor of shape (NUM_SEGMENTS*FRAMES_PER_SEGMENT) x CHANNELS x HEIGHT x WIDTH
label = sample[1]  # integer label

print('Video Tensor Size:', frame_tensor.size())

def denormalize(video_tensor):
    """
    Undoes mean/standard deviation normalization, zero to one scaling,
    and channel rearrangement for a batch of images.
    args:
        video_tensor: a (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    """
    inverse_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inverse_normalize(video_tensor) * 255.).type(torch.uint8).permute(0, 2, 3, 1).numpy()

frame_tensor = denormalize(frame_tensor)
'''
plot_video(rows=1, cols=5, frame_list=frame_tensor, plot_width=15., plot_height=3.,
           title='Evenly Sampled Frames, + Video Transform')
'''


""" DEMO 3 CONTINUED: DATALOADER """
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

for epoch in range(10):
    for video_batch, labels in dataloader:
        """
        Insert Training Code Here
        """
        print(labels)
        print("\nVideo Batch Tensor Size:", video_batch.size())
        print("Batch Labels Size:", labels.size())
        break
    break

