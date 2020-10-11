import os
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
# pip install tqdm
from tqdm import tqdm
from IPython.display import HTML, display
import time
import random
from math import pi as pi
# !pip install torch-stft
# pip install torch-stft
from torch_stft import STFT
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))
    


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            #BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        else:
            #BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


  def __init__(self, data_x, data_y, transforms=None):
    super().__init__()

    self.data_x = data_x
    self.data_y = np.array(data_y)
    self.transforms = transforms
    self.index = np.array(range(self.data_y.shape[0]))
    # print(self.index.shape)



  def __len__(self):
    return self.data_x.shape[0]
  
  def __getitem__(self, idx):
    anchor_img = self.data_x[idx]
    # print(f"Anchor Point Shape: {anchor_img.shape}")

    anchor_label = self.data_y[idx]
    # print(anchor_label)

    positive_list = self.index[self.index!=idx][self.data_y[self.index!=idx]==anchor_label]

    positive_item = random.choice(positive_list)
    positive_img = self.data_x[positive_item].reshape(126, 32, 1)
    # print(f'Positive Item Label: {self.data_y[positive_item]}')

    negative_list = self.index[self.index!=idx][self.data_y[self.index!=idx]!=anchor_label]
    negative_item = random.choice(negative_list)
    negative_img = self.data_x[negative_item].reshape(126, 32, 1)
    # print(f'Negative Item Label: {self.data_y[negative_item]}')
    anchor_img = torch.Tensor(anchor_img)
    positive_img = torch.Tensor(positive_img)
    negative_img = torch.Tensor(negative_img)

    return anchor_img.permute(2,0,1), positive_img.permute(2,0,1), negative_img.permute(2,0,1), anchor_label

def plotting_results(training_loss, validation_losses, val_targets, val_preds, train_targets, train_preds):
  plt.figure()
  plt.plot(np.arange(len(training_loss)), training_loss, label='Training Loss')
  plt.plot(np.arange(len(validation_losses)), validation_losses, label='Validation Loss')
  plt.yscale('log')
  plt.xlabel('Epoch')
  plt.show()

  fpr_val, tpr_val, _ = roc_curve(val_targets, val_preds)
  roc_auc_val = auc(fpr_val,tpr_val)

  fpr_train, tpr_train, _ = roc_curve(train_targets, train_preds)
  roc_auc_train = auc(fpr_train,tpr_train)

  plt.figure()
  lw = 2
  plt.plot(fpr_val, tpr_val, color='darkorange',
          lw=lw, label='Val ROC curve (area = %0.2f)' % roc_auc_val)
  plt.plot(fpr_train, tpr_train, color='red',
          lw=lw, label='Train ROC curve (area = %0.2f)' % roc_auc_train)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.show()


def retrieve_data(x_data_path, y_data_path, if_shuffle = False, mode=0):
    mount_path = '/content/gdrive/My Drive/nico_data'
    
    path = os.path.join(mount_path, x_data_path + '.pkl')
    with open(path, 'rb') as data:
      training_x = pickle.load(data)

    path = os.path.join(mount_path, y_data_path + '.pkl')
    with open(path, 'rb') as data:
      training_y = pickle.load(data)
      
    return_set = IQ_data(training_x, training_y)  
    return_loader = DataLoader(return_set, batch_size = 16, shuffle=if_shuffle)

    if mode == 0:
        return return_loader
    elif mode == 1:
        return training_x, training_y
     
     

def initialize_real_two(m, n):
  w_real = np.ones((1, 1, m, n))
  for i in range(m):
    for j in range(n):
      w_real[0, 0, i, j] = np.cos(2*j*pi*i/n)
  w_real = torch.from_numpy(w_real)
  w_real = w_real.type(torch.FloatTensor)
  return w_real


def initialize_imag_two(m, n):
  w_imag = np.ones((1, 1, m, n))
  for i in range(m):
    for j in range(n):
      w_imag[0, 0, i, j] = -np.sin(2*j*pi*i/n)
  w_imag = torch.from_numpy(w_imag)
  w_imag = w_imag.type(torch.FloatTensor)
  return w_imag


def initialize_real(m, n):
  w_real = np.ones((m, 1, 1, n))
  for i in range(m):
    for j in range(n):
      w_real[i, 0, 0, j] = np.cos(2*j*pi*i/n)
  w_real = torch.from_numpy(w_real)
  w_real = w_real.type(torch.FloatTensor)
  return w_real


def initialize_imag(m, n):
  w_imag = np.ones((m, 1, 1, n))
  for i in range(m):
    for j in range(n):
      w_imag[i, 0, 0, j] = -np.sin(2*j*pi*i/n)
  w_imag = torch.from_numpy(w_imag)
  w_imag = w_imag.type(torch.FloatTensor)
  return w_imag


from models.fconvnet import *
from models.MNNet import *
from utils.validation_stats import *
from datasets import *
from train import *
