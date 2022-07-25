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

import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from alive_progress import alive_bar
from torch import nn, optim
from torchvision import transforms as T, datasets, models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

import argparse

import onnx
import tensorrt as trt
import os
import torch
import torchvision
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from enum import Enum

pd.options.plotting.backend = "plotly"
from torch import nn, optim
from torch.autograd import Variable

TEST = 'test'
TRAIN = 'train'
VAL = 'val'


def data_transforms(phase=None):
  if phase == TRAIN:

    data_T = T.Compose([

      T.Resize(size=(256, 256)),
      T.RandomRotation(degrees=(-20, +20)),
      T.CenterCrop(size=224),
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  elif phase == TEST or phase == VAL:

    data_T = T.Compose([

      T.Resize(size=(224, 224)),
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

  return data_T


def build_engine_common_routine(network, builder, config, runtime, engine_file_path):

  plan = builder.build_serialized_network(network, config)
  print(f'[trace] done with builder.build_serialized_network')
  if plan == None:
    print("[trace] builder.build_serialized_network failed, exit -1")
    exit(-1)
  engine = runtime.deserialize_cuda_engine(plan)
  print("[trace] Completed creating Engine")
  with open(engine_file_path, "wb") as f:
    f.write(plan)
  return engine

class CalibratorMode(Enum):
  INT8 = 0
  FP16 = 1
  TF32 = 2
  FP32 = 3


class Calibrator(trt.IInt8EntropyCalibrator2):

  def __init__(self, training_loader, cache_file, element_bytes, batch_size=16, ):
    # Whenever you specify a custom constructor for a TensorRT class,
    # you MUST call the constructor of the parent explicitly.
    trt.IInt8EntropyCalibrator2.__init__(self)
    self.cache_file = cache_file
    self.data_provider = training_loader
    self.batch_size = batch_size
    self.current_index = 0

    # we assume single element is 4 byte
    mem_size = element_bytes * batch_size
    print(f'[trace] allocated mem_size: {mem_size}')
    self.device_input0 = cuda.mem_alloc(mem_size)
    self.device_input1 = cuda.mem_alloc(mem_size)

  def get_batch_size(self):

    return self.batch_size

  # TensorRT passes along the names of the engine bindings to the get_batch function.
  # You don't necessarily have to use them, but they can be useful to understand the order of
  # the inputs. The bindings list is expected to have the same ordering as 'names'.
  def get_batch(self, names):

    max_data_item = len(self.data_provider.dataset)
    if self.current_index + self.batch_size > max_data_item:
      return None

    _imgs0, _imgs1, labels = next(iter(self.data_provider))
    _elements0 = _imgs0.ravel().numpy()
    _elements1 = _imgs1.ravel().numpy()

    cuda.memcpy_htod(self.device_input0, _elements0)
    cuda.memcpy_htod(self.device_input1, _elements1)
    self.current_index += self.batch_size
    return [self.device_input0, self.device_input1]

  def read_calibration_cache(self):
    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
    print(f'[trace] Calibrator: read_calibration_cache: {self.cache_file}')
    if os.path.exists(self.cache_file):
      with open(self.cache_file, "rb") as f:
        return f.read()

  def write_calibration_cache(self, cache):
    print(f'[trace] Calibrator: write_calibration_cache: {cache}')
    with open(self.cache_file, "wb") as f:
      f.write(cache)


def GiB(val):
  return val * 1 << 30


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

val_famillies = "F09"
IMG_SIZE = 100
LOADER_WORER_NUMBER = 1
BATCH_SIZE = 64

def prepare_train_data():
  # An example of data:"../input/train/F00002/MID1/P0001_face1.jpg"
  all_images = glob("../data/train/*/*/*.jpg")
  train_images = [x for x in all_images if val_famillies not in x]
  val_images = [x for x in all_images if val_famillies in x]
  train_person_to_images_map = defaultdict(
    list)  # Put the link of each picture under the key word of a person such as "F0002/MID1"
  for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

  val_person_to_images_map = defaultdict(list)
  for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

  ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]
  relationships = pd.read_csv("../data/train_relationships.csv")
  relationships = list(zip(relationships.p1.values,
                           relationships.p2.values))  # For a List like[p1 p2], zip can return a result like [(p1[0],p2[0]),(p1[1],p2[1]),...]
  relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]  # filter unused relationships

  train = [x for x in relationships if val_famillies not in x[0]]
  val = [x for x in relationships if val_famillies in x[0]]

  print("Total train pairs:", len(train))
  print("Total val pairs:", len(val))

  folder_dataset = dset.ImageFolder(root='../data/train')
  print(f'[trace] define train dataset loader')
  trainset = trainingDataset(imageFolderDataset=folder_dataset,
                             relationships=train,
                             transform=transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                                           transforms.ToTensor()
                                                           ]))
  trainloader = DataLoader(trainset,
                           shuffle=True,
                           # whether randomly shuffle data in each epoch, but cannot let data in one batch in order.
                           num_workers=LOADER_WORER_NUMBER,
                           batch_size=BATCH_SIZE)

  print(f'[trace] define validation dataset loader')
  valset = trainingDataset(imageFolderDataset=folder_dataset,
                           relationships=val,
                           transform=transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                                         transforms.ToTensor()
                                                         ]))
  valloader = DataLoader(valset,
                         shuffle=True,
                         num_workers=LOADER_WORER_NUMBER,
                         batch_size=BATCH_SIZE)

  #print(f'[trace] define visual dataset loader')
  vis_dataloader = DataLoader(trainset,
                              shuffle=True,
                              num_workers=LOADER_WORER_NUMBER,
                              batch_size=8)
  dataiter = iter(vis_dataloader)

  example_batch = next(dataiter)
  concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
  #imshow(torchvision.utils.make_grid(concatenated))
  #print(example_batch[2].numpy())

  return [trainloader, valloader, vis_dataloader]

class SiameseNetwork(nn.Module):  # A simple implementation of siamese network, ResNet50 is used, and then connected by three fc layer.
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
