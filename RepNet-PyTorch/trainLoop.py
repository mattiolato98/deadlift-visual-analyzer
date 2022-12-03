import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from Model import RepNet


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ============metrics ==================


def MAE(y, ypred):
    """for period"""
    batch_size = y.shape[0]
    yarr = y.clone().detach().cpu().numpy()
    ypredarr = ypred.clone().detach().cpu().numpy()

    ae = np.sum(np.absolute(yarr - ypredarr))
    mae = ae / yarr.flatten().shape[0]
    return mae


def f1score(y, ypred):
    """for periodicity"""
    batch_size = y.shape[0]
    yarr = y.clone().detach().cpu().numpy()
    ypredarr = ypred.clone().detach().cpu().numpy().astype(bool)
    tp = np.logical_and(yarr, ypredarr).sum()
    precision = tp / (ypredarr.sum() + 1e-6)
    recall = tp / (yarr.sum() + 1e-6)
    if precision + recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    return fscore


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def getPeriodicity(periodLength):
    periodicity = torch.nn.functional.threshold(periodLength, 2, 0)
    periodicity = -torch.nn.functional.threshold(-periodicity, -1, -1)
    return periodicity


def getCount(periodLength):
    frac = 1 / periodLength
    frac = torch.nan_to_num(frac, 0, 0, 0)

    count = torch.sum(frac, dim=[1])
    return count


def getStart(periodLength):
    tmp = periodLength.squeeze(2)
    idx = torch.arange(tmp.shape[1], 0, -1)
    tmp2 = tmp * idx
    indices = torch.argmax(tmp2, 1, keepdim=True)
    return indices


def training_loop(
        epochs,
        train_set,
        val_set,
        batch_size,
        frame_per_video=64,
        lr=6e-6,
        ckpt_name='ckpt',
        use_count_error=True,
        save_ckpt=True,
        validate=True,
        checkpoint_path=None
):
    model = RepNet(frame_per_video)
    model = model.to(DEVICE)

    loss_mae = torch.nn.SmoothL1Loss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    start_epoch = 0
    train_losses = []
    val_losses = []

    if checkpoint_path:
        print('Loading checkpoint ...')

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['trainLosses']
        val_losses = checkpoint['valLosses']

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
        shuffle=True
    )

    for epoch in tqdm(range(start_epoch, epochs + start_epoch)):
        mae = 0
        mae_count = 0
        i = 1

        model.train()
        train_progress = tqdm(train_loader, total=len(train_loader))
        for inputs, labels in train_progress:
            torch.cuda.empty_cache()

            inputs = inputs.to(DEVICE).float()
            y1 = labels.to(DEVICE).float()
            y2 = getPeriodicity(y1).to(DEVICE).float()

            optimizer.zero_grad()

            outputs1, outputs2 = model(inputs)
            loss1 = loss_mae(outputs1, y1)
            loss2 = loss_bce(outputs2, y2)

            loss = loss1 + 5 * loss2

            count_prediction = torch.sum((y2pred > 0) / (y1pred + 1e-1), 1)
            count = torch.sum((y2 > 0) / (y1 + 1e-1), 1)
            loss3 = torch.sum(torch.div(torch.abs(count_prediction - count), (count + 1e-1)))

            if use_count_error:
                loss += loss3

            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            train_losses.append(train_loss)
            mae += loss1.item()
            mae_count += loss3.item()

            i += 1

            train_progress.set_postfix({
                'epoch': epoch,
                'mae_period': (mae / i),
                'mae_count': (mae_count / i),
                'mean_training_loss': np.mean(train_losses[-i + 1:]),
            })

        if validate:
            model.eval()
            with torch.no_grad():
                mae = 0
                mae_count = 0
                i = 1
                progress_bar = tqdm(val_loader, total=len(val_loader))

                for inputs, labels in progress_bar:
                    torch.cuda.empty_cache()

                    model.eval()
                    inputs = inputs.to(DEVICE).float()
                    y1 = labels.to(DEVICE).float()
                    y2 = getPeriodicity(y1).to(DEVICE).float()

                    y1pred, y2pred = model(inputs)
                    loss1 = loss_mae(y1pred, y1)
                    loss2 = loss_bce(y2pred, y2)

                    loss = loss1 + loss2

                    count_prediction = torch.sum((y2pred > 0) / (y1pred + 1e-1), 1)
                    count = torch.sum((y2 > 0) / (y1 + 1e-1), 1)
                    loss3 = loss_mae(count_prediction, count)

                    if use_count_error:
                        loss += loss3

                    val_loss = loss.item()
                    val_losses.append(val_loss)
                    mae += loss1.item()
                    mae_count += loss3.item()

                    i += 1
                    progress_bar.set_postfix({
                        'epoch': epoch,
                        'mae_period': (mae / i),
                        'mae_count': (mae_count / i),
                        'mean_validation_loss': np.mean(val_losses[-i + 1:])
                    })

        if save_ckpt:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainLosses': train_losses,
                'valLosses': val_losses
            }
            torch.save(checkpoint, 'checkpoint/' + ckpt_name + str(epoch) + '.pt')

    return train_losses, val_losses


def trainTestSplit(dataset, TTR):
    trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    valDataset = torch.utils.data.Subset(dataset, range(int(TTR * len(dataset)), len(dataset)))
    return trainDataset, valDataset


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    ave_grads = []
    max_grads = []
    median_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and (p.grad is not None) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            median_grads.append(p.grad.abs().median())

    width = 0.3
    plt.bar(np.arange(len(max_grads)), max_grads, width, color="c")
    plt.bar(np.arange(len(max_grads)) + width, ave_grads, width, color="b")
    plt.bar(np.arange(len(max_grads)) + 2 * width, median_grads, width, color='r')

    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="r", lw=4)], ['max-gradient', 'mean-gradient', 'median-gradient'])
