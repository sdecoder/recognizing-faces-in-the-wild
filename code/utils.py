from collections import defaultdict
from glob import glob

import torch
from pathlib import Path
import random
from pathlib import Path
import importlib
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader

from IPython.display import display
import pandas as pd
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from scipy.spatial.distance import euclidean
from imageio import imread
from skimage.transform import resize

# some of blocks below are not used.

# Data manipulation
import numpy as np
import pandas as pd

# Data visualisation
import matplotlib.pyplot as plt

# Fastai
# from fastai.vision import *
# from fastai.vision.models import *

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils
import torchvision.datasets as dset

from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import *
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
# import pretrainedmodels

from pathlib import Path
import sys


class trainingDataset(Dataset):  # Get two images and whether they are related.

  def __init__(self, imageFolderDataset, relationships, transform=None):

    self.imageFolderDataset = imageFolderDataset
    self.relationships = relationships  # choose either train or val dataset to use
    self.transform = transform

  def __getitem__(self, index):
    img0_info = self.relationships[index][0]
    # for each relationship in train_relationships.csv, the first img comes from first row, and the second is either specially choosed related person or randomly choosed non-related person
    img0_path = glob("../data/train/" + img0_info + "/*.jpg")
    img0_path = random.choice(img0_path)

    cand_relationships = [x for x in self.relationships if
                          x[0] == img0_info or x[1] == img0_info]  # found all candidates related to person in img0
    if cand_relationships == []:  # in case no relationship is mensioned. But it is useless here because I choose the first person line by line.
      should_get_same_class = 0
    else:
      should_get_same_class = random.randint(0, 1)

    if should_get_same_class == 1:  # 1 means related, and 0 means non-related.
      img1_info = random.choice(cand_relationships)  # choose the second person from related relationships
      if img1_info[0] != img0_info:
        img1_info = img1_info[0]
      else:
        img1_info = img1_info[1]
      img1_path = glob("../data/train/" + img1_info + "/*.jpg")  # randomly choose a img of this person
      img1_path = random.choice(img1_path)
    else:  # 0 means non-related
      randChoose = True  # in case the chosen person is related to first person
      while randChoose:
        img1_path = random.choice(self.imageFolderDataset.imgs)[0]
        img1_info = img1_path.split("/")[-3] + "/" + img1_path.split("/")[-2]
        randChoose = False
        for x in cand_relationships:  # if so, randomly choose another person
          if x[0] == img1_info or x[1] == img1_info:
            randChoose = True
            break

    img0 = Image.open(img0_path)
    img1 = Image.open(img1_path)

    if self.transform is not None:  # I think the transform is essential if you want to use GPU, because you have to trans data to tensor first.
      img0 = self.transform(img0)
      img1 = self.transform(img1)

    return img0, img1, should_get_same_class  # the returned data from dataloader is img=[batch_size,channels,width,length], should_get_same_class=[batch_size,label]

  def __len__(self):
    return len(self.relationships)  # essential for choose the num of data in one epoch

class SiameseNetwork(
  nn.Module):  # A simple implementation of siamese network, ResNet50 is used, and then connected by three fc layer.
  def __init__(self):
    super(SiameseNetwork, self).__init__()
    # self.cnn1 = models.resnet50(pretrained=True)#resnet50 doesn't work, might because pretrained model recognize all faces as the same.
    self.cnn1 = nn.Sequential(
      nn.ReflectionPad2d(1),
      nn.Conv2d(3, 64, kernel_size=3),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(64),
      nn.Dropout2d(p=.2),

      nn.ReflectionPad2d(1),
      nn.Conv2d(64, 64, kernel_size=3),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(64),
      nn.Dropout2d(p=.2),

      nn.ReflectionPad2d(1),
      nn.Conv2d(64, 32, kernel_size=3),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(32),
      nn.Dropout2d(p=.2),
    )
    self.fc1 = nn.Linear(2 * 32 * 100 * 100, 500)
    # self.fc1 = nn.Linear(2*1000, 500)
    self.fc2 = nn.Linear(500, 500)
    self.fc3 = nn.Linear(500, 2)

  def forward(self, input1, input2):  # did not know how to let two resnet share the same param.
    output1 = self.cnn1(input1)
    output1 = output1.view(output1.size()[0], -1)  # make it suitable for fc layer.
    output2 = self.cnn1(input2)
    output2 = output2.view(output2.size()[0], -1)

    output = torch.cat((output1, output2), 1)
    output = F.relu(self.fc1(output))
    output = F.relu(self.fc2(output))
    output = self.fc3(output)
    return output
