import os
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import transforms as T
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm, Flatten, Conv2d
import torchvision
from torchvision import transforms, models
import math
import copy
import torchsummary
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
import GPUtil
import nibabel as nib
import numpy as np
from tqdm import tqdm, trange
from itertools import cycle
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm
from math import exp, sqrt
from PIL import Image
import wandb
from torchvision.utils import save_image
import torchvision.transforms.functional as FF
import lpips



parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default="3")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=101)
parser.add_argument('--lr_age', type=float, default=0.0005)
parser.add_argument('--lr_gan', type=float, default=0.0005)
parser.add_argument('--lr_map', type=float, default=0.00001)
parser.add_argument('--id_optim', type=str, default="Adam")
args = parser.parse_args()



def age_onehot(age):
    if age.dim() == 0:
        age = age.view(1)

    z = torch.zeros(len(age), 101)
    z[torch.arange(len(age)), age] = 1
    return z

