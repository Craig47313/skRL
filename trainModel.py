# learned from and used some modified code from Rob Mulla's 'Train Your first PyTorch Model [Card Classifier]' which can be found at https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier
# --> also learned using Kie Codes 'Image Classifier in PyTorch' which can be found at https://www.youtube.com/watch?v=igQeI29FIQM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)