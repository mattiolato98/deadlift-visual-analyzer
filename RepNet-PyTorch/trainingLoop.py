import torch
import random

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

'''
deadlift_dataset = getCombinedDataset('deadlift_videos/repnet.csv',
                                   'deadlift_videos',
                                   'Good')

dataset_size = len(deadlift_dataset)
val_size = int(np.floor(0.2 * dataset_size))
train_size = dataset_size - val_size
train_set, val_set = torch.utils.data.random_split(deadlift_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))




# testDatasetC = getCombinedDataset('countix/countix_test.csv',
#                                    'testvids',
#                                    'test')

# testDatasetC = getCombinedDataset('deadlift_downscaled_360p/repnet.csv',
#                                    'deadlift_downscaled_360p',
#                                    'Good')
testDatasetS = SyntheticDataset('synthvids', 'train*', 'mp4', 2000)

testList = [testDatasetC] #, testDatasetS]
random.shuffle(testList)
testDataset = ConcatDataset(testList)



# trainDatasetC = getCombinedDataset('countix/countix_train.csv',
#                                    'trainvids',
#                                    'train')
trainDatasetC = getCombinedDataset('deadlift_videos/repnet.csv',
                                   'deadlift_videos',
                                   'Good')
#trainDatasetS1 = SyntheticDataset('/home/saurabh/Downloads/HP72','HP72', 'mp4', 500,
#                                   frame_per_vid=frame_per_vid)
#trainDatasetS2 = SyntheticDataset('/home/saurabh/Downloads', '1917', 'mkv', 500,
#                                   frame_per_vid=frame_per_vid)

'''

trainDatasetS3 = SyntheticDataset('synthvids', '*', 'mp4', 3000)
#trainDatasetS4 = SyntheticDataset('/home/saurabh/Downloads', 'HP6', 'mkv', 500,
#                                   frame_per_vid=frame_per_vid)
trainDatasetB = BlenderDataset('blendervids', 'videos', 'annotations', frame_per_vid)

trainList = [trainDatasetS3] #, trainDatasetB]
random.shuffle(trainList)
trainDataset = ConcatDataset(trainList)

print("done")

"""Testing the training loop with sample datasets"""

sampleDatasetA = torch.utils.data.Subset(trainDataset, range(0, len(trainDataset)))
# sampleDatasetB = torch.utils.data.Subset(testDataset, range(0,  len(testDataset)))

print(len(sampleDatasetA))

trLoss, valLoss = training_loop(
    50,
    sampleDatasetA,
    sampleDatasetA,
    1,
    lr=6e-5,
    ckpt_name='x3dbb',
    use_count_error=True,
    checkpoint_path='checkpoint/bce_mae_count_one_tr9.pt'
)
