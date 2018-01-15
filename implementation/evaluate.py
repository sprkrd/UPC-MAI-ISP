import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from skimage.transform import resize

from data.EuroNotes import EuroNotes

from models.ResNet18 import pretrained_res18

import numpy as np

import time


# means and stds of the train set can be obtained with 
# print(train_set.getMeansAndStdPerChannel())
# this takes quite a while, so hard-coded here:
# means = np.array([ 0.14588552,  0.26887908,  0.14538361])
# stds = np.array([ 0.20122388,  0.2800698 ,  0.20029236])
means = np.array([ 0.34065133, 0.30230788, 0.27947797])
stds = np.array([ 0.28919015, 0.26877816, 0.25182973])


def resize_transform(img):
    return resize(img, (224, 224), preserve_range=True, mode="constant")

transformations = transforms.Compose([resize_transform, transforms.ToTensor(), transforms.Normalize(means, stds)])

val_set = EuroNotes('../data-augmentation/banknotes_augmented/test', transform=transformations, resize=False)

val_loader = DataLoader(val_set, batch_size=25, shuffle=True, num_workers=2)

cnn = pretrained_res18()

sampleBatches = 100

correct = 0.0   
total = 0.0
for i_batch, data in enumerate(val_loader):
    if i_batch > sampleBatches:
            break
    images = Variable(data['image'], volatile=True)
    labels = data['label'].type(torch.LongTensor)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.type(torch.LongTensor)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    validationAcc = 100 * correct / total
    print('{}/{} Test Accuracy of CNN: {:.02f}%'.format(i_batch+1, sampleBatches, validationAcc), end="\r")


