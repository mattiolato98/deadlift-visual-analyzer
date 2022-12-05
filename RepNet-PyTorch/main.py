import os
import time

import torch
import random
import numpy as np

from torch.utils.data import DataLoader, ConcatDataset, RandomSampler


from trainLoop import running_mean, training_loop, trainTestSplit, plot_grad_flow
# from Model_inn2 import RepNet
from Model import RepNet
from Dataset import getCombinedDataset
from SyntheticDataset import SyntheticDataset
from BlenderDataset import BlenderDataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cpu")

frame_per_vid = 64
multiple = False

if __name__ == '__main__':
    '''        
    # testDatasetC = getCombinedDataset('countix/countix_test.csv',
    #                                    'testvids',
    #                                    'test')
    

    
    testList = [testDatasetC] #, testDatasetS]
    random.shuffle(testList)
    testDataset = ConcatDataset(testList)
    
    
    
    # trainDatasetC = getCombinedDataset('countix/countix_train.csv',
    #                                    'trainvids',
    #                                    'train')

    #trainDatasetS2 = SyntheticDataset('/home/saurabh/Downloads', '1917', 'mkv', 500,
    #                                   frame_per_vid=frame_per_vid)
    
    '''

    #trainDatasetS4 = SyntheticDataset('/home/saurabh/Downloads', 'HP6', 'mkv', 500,
    #                                   frame_per_vid=frame_per_vid)
    # trainDatasetB = BlenderDataset('blendervids', 'videos', 'annotations', frame_per_vid)

    # testDatasetC = getCombinedDataset('deadlift_downscaled_360p/repnet.csv',
    #                                   'deadlift_downscaled_360p',
    #                                   'Good')
    # testDatasetS = SyntheticDataset('synthvids', 'train*', 'mp4', 2000)
    #
    # trainDatasetS3 = SyntheticDataset('synthvids', '*', 'mp4', 3000)
    # trainDatasetC = getCombinedDataset('deadlift_videos/repnet.csv',
    #                                    'deadlift_videos',
    #                                    'Good')

    print(f'CARTELLA: {os.listdir()}\n\n')
    time.sleep(1000)

    dataset = getCombinedDataset(
        'deadlift_downscaled_540p/repnet.csv',
        'deadlift_downscaled_540p_reps',
        'Good'
    )
    synthetic_dataset = SyntheticDataset('deadlift_downscaled_540p_reps/', '*', 'mp4', 3000)

    dataset_size = len(dataset)
    val_size = int(np.floor(0.2 * dataset_size))
    train_size = dataset_size - val_size
    comb_train_set, comb_val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    synthetic_dataset_size = len(synthetic_dataset)
    synthetic_val_size = int(np.floor(0.2 * synthetic_dataset_size))
    synthetic_train_size = synthetic_dataset_size - synthetic_val_size
    synthetic_train_set, synthetic_val_set = torch.utils.data.random_split(
        synthetic_dataset, [synthetic_train_size, synthetic_val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset_list = [comb_train_set, synthetic_train_set]
    random.shuffle(train_dataset_list)
    train_set = ConcatDataset(train_dataset_list)

    val_dataset_list = [comb_val_set, synthetic_val_set]
    random.shuffle(val_dataset_list)
    val_set = ConcatDataset(val_dataset_list)

    # sampleDatasetA = torch.utils.data.Subset(trainDataset, range(0, len(trainDataset)))
    # sampleDatasetB = torch.utils.data.Subset(testDataset, range(0,  len(testDataset)))

    trLoss, valLoss = training_loop(
        50,
        train_set,
        val_set,
        1,
        lr=6e-5,
        ckpt_name='x3dbb',
        use_count_error=True,
        checkpoint_path='checkpoint/bce_mae_count_one_tr9.pt'
    )
