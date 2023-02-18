import torch
import os
import time
import copy
import numpy as np
from torchvision import datasets
import pandas as pd

from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    CenterCropVideo,
    RandomCrop

)

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    RandomShortSideScale,
    ShortSideScale
)
from typing import Dict

import torch.optim as optim
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import RandomSampler, SequentialSampler
import pickle

####################
# SlowFast transform
####################


model_name = "slowfast_r101"
num_classes = 2
feature_extract = True

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
alpha = 4

weights = 'custom_weights/slowfast_final_weights.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PackPathway(torch.nn.Module):
    """A module for transforming video frames into a list of tensors representing different pathways.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            RandomShortSideScale(min_size=256, max_size=320),
            CenterCropVideo(224),
            RandomHorizontalFlip(p=0.5),
            PackPathway()
        ]
    ),
)

test_transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway(),
        ]
    ),
)

'''
####################
# Slow transform
####################

side_size = 224
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 224
num_frames = 8
sampling_rate = 8
frames_per_second = 30

# Note that this transform is specific to the slow_R50 model.
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            RandomShortSideScale(min_size=256, max_size=320),
            RandomCrop(224),
            RandomHorizontalFlip(p=0.5)
        ]
    ),
)
'''
# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate) / frames_per_second


class DeadliftDataset(datasets.VisionDataset):
    """A dataset class for loading deadlift videos.

    Args:
        csv_file (str): Path to the CSV file containing video paths and labels.
        **kwargs: Additional keyword arguments to be passed to the parent class.

    Attributes:
        videos (pd.DataFrame): A DataFrame containing the video paths and labels.
        root_dir (str): The root directory containing the video files.
        transform (callable): A function/transform to be applied on the video data.

    """

    def __init__(self, csv_file, **kwargs):
        super().__init__(**kwargs)
        self.videos = pd.read_csv(csv_file)
        self.root_dir = kwargs['root']
        self.transform = kwargs['transform']

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_path = os.path.join(self.root_dir, self.videos.iloc[idx, 0])

        start_sec = 0
        end_sec = start_sec + clip_duration

        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

        label = self.videos.iloc[idx, 1]
        if self.transform:
            # print("Transforming ...")
            video_data = self.transform(video_data)
            # print("Transformation done")

        # sample = {'video' : video_data, 'label' : label }

        return video_data["video"], label


def set_parameter_requires_grad(model, feature_extracting):
    """Sets requires_grad attribute of a model's parameters based on feature_extracting flag.

    Args:
        model (nn.Module): Model to set requires_grad attribute for.
        feature_extracting (bool): Flag indicating whether to set requires_grad to False for all parameters.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def available_models():
    """Print available pretrained pytorchvideo models.
    """
    entrypoints = torch.hub.list('facebookresearch/pytorchvideo', force_reload=True)
    print('Available pretrained pytorchvideo models:')
    for model in entrypoints:
        print(model)


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """Initializes a model with a specified architecture and number of output classes.

    Args:
        model_name (str): Name of the model architecture.
        num_classes (int): Number of output classes for the model.
        feature_extract (bool): Whether to freeze the weights of the model or not.
        use_pretrained (bool): Whether to use pre-trained weights or not.

    Returns:
        A model with the specified architecture and number of output classes.
    """
    model_ft = None

    if model_name == "slowfast_r50":
        """ Slowfast_R50 
        """
        model_ft = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.blocks[6].proj.in_features
        model_ft.blocks[6].proj = nn.Linear(num_ftrs, num_classes)


    elif model_name == "slow_r50":
        """ Slow_R50
        """
        model_ft = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.blocks[5].proj.in_features
        model_ft.blocks[5].proj = nn.Linear(num_ftrs, num_classes)

    elif model_name == "slowfast_r101":
        """ Slowfast_R101
                """
        model_ft = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.blocks[6].proj.in_features
        model_ft.blocks[6].proj = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


def train_model_v2(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """Train the given model for the given number of epochs using the specified train and validation data loaders.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_loader (torch.utils.data.DataLoader): The data loader for the training data.
            val_loader (torch.utils.data.DataLoader): The data loader for the validation data.
            criterion (torch.nn.modules.loss._Loss): The loss function to be used during training.
            optimizer (torch.optim.Optimizer): The optimizer to be used during training.
            num_epochs (int): The number of epochs to train the model.

        Returns:
            tuple: A tuple containing the trained model, training loss values, training accuracy values,
            validation loss values, and validation accuracy values.
        """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_values = []
    val_loss_values = []
    train_acc_values = []
    val_acc_values = []
    epoch_count = []

    for epoch in range(num_epochs):
        print(f"Epoch : {epoch} of {num_epochs}")

        # Each epoch has a training and validation phase

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            # print(f"New training batch -> Inputs shape : {inputs.shape}, labels shape : {labels.shape}")
            # inputs = inputs.to(device)
            inputs = [i.to(device) for i in inputs]
            labels = labels.to(device)
            # print(f"Shape inputs {inputs.shape}")
            # print(f"Shape labels {labels.shape}")
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                # print(f"Preds : {preds}")
                # Statistics

                running_loss += loss.item() * inputs[0].size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f"Epoch loss (mean): {epoch_loss}, Epoch acc : {epoch_acc}")

        # Evaluation
        print(f"Evaluation of epoch {epoch} of {num_epochs}")
        model.eval()
        running_loss_val = 0.0
        running_corrects_val = 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                # print(f"New evaluation batch -> Inputs shape : {inputs.shape}, labels shape : {labels.shape}")
                # inputs = inputs.to(device)
                inputs = [i.to(device) for i in inputs]
                labels = labels.to(device)
                outputs = model(inputs)
                loss_val = criterion(outputs, labels)

                running_loss_val += loss_val.item() * inputs[0].size(0)
                _, preds_val = torch.max(outputs, 1)
                print(f"Real labels : {labels} ")
                print(f"Predictions : {preds_val} \n")
                running_corrects_val += torch.sum(preds_val == labels)

            epoch_loss_val = running_loss_val / len(val_loader.dataset)
            epoch_acc_val = running_corrects_val / len(val_loader.dataset)
            print(f"Epoch val loss (mean): {epoch_loss_val} item {loss_val.item()}, Epoch val acc : {epoch_acc_val}")

            if epoch_acc_val > best_acc:
                best_acc = epoch_acc_val
                best_model_wts = copy.deepcopy(model.state_dict())

            train_acc_values.append(epoch_acc)
            val_acc_values.append(epoch_acc_val)

            train_loss_values.append(epoch_loss)
            val_loss_values.append(epoch_loss_val)
            epoch_count.append(epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    print(f"Saving model parameters to {model_save_path}")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': best_model_wts,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss_values[-1],
    }, model_save_path)
    print("Parameters successfully saved")

    return model, train_loss_values, train_acc_values, val_loss_values, val_acc_values


def split_dataset(dataset_name, batch_size=8, transform=None):
    """Returns train and validation data loaders for the given dataset.

    Args:
        dataset_name (str): The name of the dataset.
        batch_size (int): The batch size
        transform: Transformations to apply to the dataset.

    Returns:
        tuple: A tuple containing the train and validation data loaders.
    """

    dataset = Path(os.getcwd()) / f"{dataset_name}"
    deadlift_dataset = DeadliftDataset(root=dataset, csv_file=dataset / "dataset.csv", transform=transform)

    dataset_size = len(deadlift_dataset)

    '''
    # Another type of train/val split
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)
    val_split_index = int(np.floor(0.2 * dataset_size))
    train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]
    
    train_sampler = SubsetRandomSampler(train_idx)
    # val_sampler = SubsetRandomSampler(val_idx)
    val_sampler = SequentialSampler(val_idx)
    '''
    val_size = int(np.floor(0.2 * dataset_size))
    train_size = dataset_size - val_size
    train_set, val_set = torch.utils.data.random_split(deadlift_dataset, [train_size, val_size],
                                                       generator=torch.Generator().manual_seed(42))
    train_sampler = RandomSampler(train_set, replacement=True)

    train_loader = torch.utils.data.DataLoader(
        batch_size=batch_size,
        dataset=train_set,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True

    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    return train_loader, val_loader


def save_training_stats(train_loss_values, train_acc_values, val_loss_values, val_acc_values, saving_model_name):
    """Save the training statistics to a pickle file.

    Args:
        train_loss_values (list): List of training losses for each epoch.
        train_acc_values (list): List of training accuracies for each epoch.
        val_loss_values (list): List of validation losses for each epoch.
        val_acc_values (list): List of validation accuracies for each epoch.
        saving_model_name (str): Name of the model being saved.
    """
    print("Saving training stats....")
    path = Path(os.getcwd()) / f"Deadlift_models/{saving_model_name}.pickle"
    save_object = {'loss': (train_loss_values, val_loss_values), 'acc': (train_acc_values, val_acc_values)}
    with open(path, 'wb') as handle:
        pickle.dump(save_object, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Training statistics successfully saved")


def inference(video_path, reps_range):
    """Evaluate the number of deadlift repetitions in a given video.

    Args:
        video_path (str): File path to the input video.
        reps_range (list): List of tuples indicating start and end times (in seconds) of each repetition in the video.

    Returns:
        list: A list of predicted classes for each repetition, where 0 indicates a bad repetition and 1 indicates a good
         repetition.
    """
    print("Evaluation of your repetitions started")
    model = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # Initialize an EncodedVideo helper class
    video = EncodedVideo.from_path(video_path)

    batch_slow = torch.empty(0)
    batch_slow = batch_slow.to(device)
    batch_fast = torch.empty(0)
    batch_fast = batch_fast.to(device)

    for start_sec, end_sec in reps_range:
        # Load the desired clip
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

        # Apply a transform to normalize the video input
        video_data = test_transform(video_data)

        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = [i.to(device) for i in inputs]

        batch_slow = torch.cat((batch_slow, inputs[0].unsqueeze(0)), 0)
        batch_fast = torch.cat((batch_fast, inputs[1].unsqueeze(0)), 0)

    final_batch = [batch_slow, batch_fast]

    model.eval()

    post_act = torch.nn.Sigmoid()
    preds = model(final_batch)
    preds = post_act(preds)
    pred_classes = preds.topk(k=1).indices

    # From tensor to list
    pred_classes = pred_classes.tolist()

    # From list of lists to single list
    final_predictions = []
    for elem in pred_classes:
        if type(elem) is list:
            for item in elem:
                final_predictions.append(item)
        else:
            final_predictions.append(elem)
    print("Assessment of your repetitions successfully completed")
    # final_predictions = [item in sublist for sublist in pred_classes for item in sublist]
    return final_predictions


def evaluate_accuracy():
    """Evaluate the accuracy of the model on the test set.

    Loads the test set from a CSV file and applies the test_transform
    to normalize the video input. Then it loads the model from the saved
    checkpoint and applies it to the inputs of the test set, calculating
    the running_corrects metric to obtain the total accuracy on the test set.

    """
    test_dataset = DeadliftDataset(root=dataset, csv_file=dataset / "test.csv", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Evaluation
    print(f"Accuracy test of the model")
    model = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    model.load_state_dict(torch.load(Path(os.getcwd()) / weights, map_location=device))
    model.to(device)
    model.eval()
    running_corrects = 0

    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = [i.to(device) for i in inputs]
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds_val = torch.max(outputs, 1)
            print(f"Real labels : {labels} ")
            print(f"Predictions : {preds_val} \n")
            running_corrects += torch.sum(preds_val == labels)

        accuracy = running_corrects / len(test_dataset)
        print(f"Total accuracy on test set: {accuracy}")


if __name__ == '__main__':
    project_path = Path(os.getcwd())
    num_epochs = 200
    model_path = project_path / "Deadlift_models/"
    saving_model_name = "final_training_200ep"
    loading_model_name = "centercrop_newtr_100ep_noinitilaframes"

    # Initialize the model for this run
    resume = False

    if not resume:
        model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    else:
        model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
        model_path.mkdir(parents=True, exist_ok=True)
        model_load_name = f"{loading_model_name}.pth"
        model_load_path = model_path / model_load_name

    # Gather the parameters to be optimized/updated in this run.
    # If we are finetuning we will be updating all parameters. However, if we are
    # doing feature extract method, we will only update the parameters
    # that we have just initialized, i.e. the parameters with requires_grad is True.
    model_ft.to(device)
    params_to_update = model_ft.parameters()
    print(f"{model_name} initialization completed")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    if resume:
        checkpoint = torch.load(model_load_path)
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epochs = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Custom weights successfully loaded")

    model_save_name = f"{saving_model_name}.pth"
    model_save_path = model_path / model_save_name

    dataset = "Dataset_downscaled_540p"
    train_loader, val_loader = split_dataset(dataset, batch_size=16, transform=transform)

    model_ft, tr_l, tr_a, val_l, val_a = train_model_v2(model_ft, train_loader, val_loader, loss_fn, optimizer_ft,
                                                        num_epochs=num_epochs)

    save_training_stats(tr_l, tr_a, val_l, val_a, saving_model_name)

# inference()
