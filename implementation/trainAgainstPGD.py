# so it can be executed both with Python2 and Python3
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import os
import numpy as np
import time

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.transform import resize

from .data.EuroNotes import EuroNotes
from .utils import means, stds
from .attackers.WhiteBoxAttacker import PGDAttack


##
# Run on GPU?
##
runGPU = True
if runGPU:
    cudnn.benchmark = True


transformations = transforms.Compose([
    lambda img: resize(img, (224,224), preserve_range=True, mode="constant"),
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
])

train_set = EuroNotes('data-augmentation/banknotes_augmented/train', transform=transformations, resize=False)
val_set = EuroNotes('data-augmentation/banknotes_augmented/val', transform=transformations, resize=False)


train_loader = DataLoader(train_set, batch_size=25, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=25, shuffle=True, num_workers=2)

cnn = torchvision.models.resnet18(pretrained=True)
# for param in cnn.parameters():
#       param.requires_grad = False
cnn.fc = nn.Linear(512, 4)

if runGPU:
    cnn.cuda()
else:
    cnn.cpu()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters())


epochs = 10

bestValidationAcc = 0.0

# attack = PGDAttack(k=5)
attack = PGDAttack(k=10, epsilon=0.05)

for i in range(0, epochs):
    losses = 0
    lossSum = 0.0
    t = time.time()
    for i_batch, sample_batched in enumerate(train_loader):
        images = sample_batched['image']
        labels = sample_batched['label']
        if runGPU:
            images = images.cuda()
            labels = labels.cuda()
        # the attack method does not modify the model
        if cnn.training:
            cnn.eval()
        # the attack method returns the perturbed images already encapsulated
        # in a variable
        images_pert = attack.attack(cnn, images, labels)
        labels = Variable(labels)
        cnn.train()
        optimizer.zero_grad()
        outputs = cnn(images_pert)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses += 1
        lossSum += loss.data[0]
        batchesPerReport = 100
        if i_batch % batchesPerReport == 0:
            print('> batch ' + str(i_batch) + ', current average loss: ' + str(lossSum / losses))
            losses = 0
            lossSum = 0
    
    if cnn.training:
        cnn.eval()
    
    sampleBatches = 100

    print("Evaluating against training set...")
    correct = 0.0   
    total = 0.0
    for i_batch, data in enumerate(train_loader):
        if i_batch > sampleBatches:
            break
        images = sample_batched['image']
        labels = sample_batched['label'].type(torch.LongTensor)
        if runGPU:
            images = images.cuda()
        images_pert = attack.attack(cnn, images, labels)
        outputs = cnn(images_pert)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.type(torch.LongTensor)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    trainAcc = correct / total

    print("Evaluating against test set...")
    correct = 0.0
    total = 0.0
    for i_batch, data in enumerate(val_loader):
        if i_batch > sampleBatches:
            break
        images = sample_batched['image']
        labels = sample_batched['label'].type(torch.LongTensor)
        if runGPU:
            images = images.cuda()
        images_pert = attack.attack(cnn, images, labels)
        outputs = cnn(images_pert)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.type(torch.LongTensor)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    validationAcc = correct / total
    print('-- EPOCH ' + str(i+1) + ' DONE.')
    print('Time elapsed: ' + str(time.time() - t))
    print('Sampled accuracies from ' + str(sampleBatches) + ' batches.')
    print('(sampled) Attacked Train Accuracy of CNN: ' + str(trainAcc * 100) + '%')
    print('(sampled) Attacked Validation Accuracy of CNN: ' + str(validationAcc * 100) + '%')
    print('--')

    if validationAcc > bestValidationAcc:
        try:
            os.mkdir("out")
        except OSError:
            pass
        bestValidationAcc = validationAcc
        torch.save(cnn.state_dict(), 'out/baseClassifier_best.pkl')
        print('Stored new best base classifier')
        print('--')

