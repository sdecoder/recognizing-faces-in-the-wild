from collections import defaultdict

import torch
from pathlib import Path
import random
from pathlib import Path
import importlib

from alive_progress import alive_bar
from apex import amp
from apex.amp.frontend import O3
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

from glob import glob
from PIL import Image

from utility import trainingDataset, SiameseNetwork, prepare_train_data

NUMBER_EPOCHS = 50


def imshow(img, text=None, should_save=False):  # for showing the data you loaded to dataloader
  npimg = img.numpy()
  plt.axis("off")
  if text:
    plt.text(75, 8, text, style='italic', fontweight='bold',
             bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


def show_plot(iteration, loss):  # for showing loss value changed with iter
  plt.plot(iteration, loss)
  plt.savefig('../resources/acc_train_wo_amp.png')
  plt.show()

# F09xx are used for validation.
def train(trainloader, valloader, vis_dataloader):

  import utility
  print(f'[trace] exec at the train func[w/o apex]')
  model = SiameseNetwork().cuda()
  criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  #prepare for the apex training
  opt_level = 'O3'
  #model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

  counter = []
  loss_history = []
  iteration_number = 0

  best_accuracy = -float('inf')
  print(f'[trace] exec in the train function')
  import time
  time_epochs = []
  for epoch in range(0, NUMBER_EPOCHS):
    print("Epoch：", epoch, " start.")
    with alive_bar(len(trainloader), dual_line=True, title='trainloader') as bar:

      this_time_epoch = 0
      for i, data in enumerate(trainloader, 0):
        bar.text = f'[trace] epoch {epoch} index {i}/{len(trainloader)}: training'
        bar()

        img0, img1, labels = data  # img=tensor[batch_size,channels,width,length], label=tensor[batch_size,label]
        img0, img1, labels = img0.cuda(), img1.cuda(), labels.cuda()  # move to GPU
        # print("epoch：", epoch, "No." , i, "th inputs", img0.data.size(), "labels", labels.data.size())

        start_epoch = time.time()
        optimizer.zero_grad()  # clear the calculated grad in previous batch

        '''
        # debug purpose
        print(f'[trace] torch.size: img0: {img0.size()}')
        print(f'[trace] torch.size: img1: {img1.size()}')
        exit(0)
        '''

        outputs = model(img0, img1)
        loss = criterion(outputs, labels)

        #with amp.scale_loss(loss, optimizer) as scaled_loss:
        #  scaled_loss.backward()
        loss.backward()
        optimizer.step()

        end_epoch = time.time()
        this_time_epoch += end_epoch - start_epoch

        if i % 10 == 0:  # show changes of loss value after each 10 batches
          # print("Epoch number {}\n Current loss {}\n".format(epoch,loss.item()))
          iteration_number += 10
          counter.append(iteration_number)
          loss_history.append(loss.item())


    print(f'[trace] time used in this epoch {this_time_epoch} seconds')
    time_epochs.append(this_time_epoch)

    # test the network after finish each epoch, to have a brief training result.
    print(f'[trace] start to validate')
    correct_val = 0
    total_val = 0
    with torch.no_grad():  # essential for testing!!!!
      with alive_bar(len(valloader), dual_line=True, title='valloader') as bar:
        for i, data in enumerate(valloader):
          bar.text = f'[trace] epoch {epoch} index {i}/{len(valloader)}: validating'
          bar()

          img0, img1, labels = data
          img0, img1, labels = img0.cuda(), img1.cuda(), labels.cuda()
          outputs = model(img0, img1)
          _, predicted = torch.max(outputs.data, 1)
          total_val += labels.size(0)
          correct_val += (predicted == labels).sum().item()

    accuracy = (100 * correct_val / total_val)
    print('Accuracy of the network on the', total_val, 'val pairs in', utility.val_famillies,
          ': %d %%' % (100 * correct_val / total_val))
    show_plot(counter, loss_history)
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      path = f'../models/net_epoch_{epoch}.pt'
      print(f'[trace] better model found, saving model to {path}')
      torch.save(model.state_dict(), path)

  print(f'[trace] time used in each epoch: {time_epochs}')
  print(f'[trace] end of the train function')
  pass

def main():
  print(f'[trace] exec main function')
  [trainloader, valloader, vis_dataloader] = prepare_train_data()
  train(trainloader, valloader, vis_dataloader)
  print(f'[trace] done with main function')
  pass


if __name__ == '__main__':
  main()
  pass
