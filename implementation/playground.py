import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from data.EuroNotes import EuroNotes

from models.BasicCNN import BasicCNN

import time

import numpy as np


# print(train_set.getMeansAndStdPerChannel())

# means and stds of the train set can be obtained with 
# print(train_set.getMeansAndStdPerChannel())
# this takes quite a while, so hard-coded here:
means = np.array([ 0.14588552,  0.26887908,  0.14538361])
stds = np.array([ 0.20122388,  0.2800698 ,  0.20029236])

transformations = transforms.Compose([transforms.RandomCrop(256), transforms.ToTensor(), transforms.Normalize(means, stds)])


train_set = EuroNotes('../data-augmentation/banknotes_augmented/test', transform=transformations)
# val_set = EuroNotes('../banknotes_augmented/val', transform=to_tensor)

train_loader = DataLoader(train_set, batch_size=100, shuffle=True, num_workers=16)


for i_batch, sample_batched in enumerate(train_loader):

	images = Variable(sample_batched['image'])

	print(images[0][0,0:20,0:20])
	break