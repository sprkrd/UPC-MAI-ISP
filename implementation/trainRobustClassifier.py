import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import time 
import numpy as np

from data.EuroNotes import EuroNotes

from models.ResNet18 import pretrained_res18

from attackers.WhiteBoxAttacker import PGDAttack

##
# Run on GPU?
##
runGPU = False
if runGPU:
  cudnn.benchmark = True


means = np.array([ 0.34065133, 0.30230788, 0.27947797])
stds = np.array([ 0.28919015, 0.26877816, 0.25182973])




transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, stds)])

train_set = EuroNotes('data-augmentation/banknotes_augmented/train', transform=transformations, resize=False)
val_set = EuroNotes('data-augmentation/banknotes_augmented/val', transform=transformations, resize=False)


train_loader = DataLoader(train_set, batch_size=25, shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size=25, shuffle=True, num_workers=16)

model = pretrained_res18()

if runGPU:
  model.cuda()
else:
  model.cpu()

model.train()

attacker = PGDAttack()




criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs = 100

for i in range(0, epochs):
  model.train()

  losses = 0
  lossSum = 0.0
  t = time.time()
  for i_batch, sample_batched in enumerate(train_loader):

    images = sample_batched['image']
    labels = sample_batched['label']

    # attack the images:
    images, _ = attacker.attack(model, images, labels)
    
    labels = Variable(labels)


    if runGPU:
      images = images.cuda()
      labels = labels.cuda()

    optimizer.zero_grad()

    outputs = model(images)

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
  
  model.eval()
  
  # sampleBatches = 100

  # correct = 0.0  
  # total = 0.0
  # for i_batch, data in enumerate(train_loader):
  #     if i_batch > sampleBatches:
  #         break
  #     images = Variable(data['image'], volatile=True)
  #     labels = data['label'].type(torch.LongTensor)
  #     if runGPU:
  #         images = images.cuda()
  #     outputs = model(images)
  #     _, predicted = torch.max(outputs.data, 1)
  #     predicted = predicted.type(torch.LongTensor)
  #     total += labels.size(0)
  #     correct += (predicted == labels).sum()
  # trainAcc = correct / total

  # correct = 0.0
  # total = 0.0
  # for i_batch, data in enumerate(val_loader):
  #   if i_batch > sampleBatches:
  #     break
  #   images = Variable(data['image'], volatile=True)
  #   labels = data['label'].type(torch.LongTensor)
  #   if runGPU:
  #     images = images.cuda()
  #   outputs = model(images)
  #   _, predicted = torch.max(outputs.data, 1)
  #   predicted = predicted.type(torch.LongTensor)
  #   total += labels.size(0)
  #   correct += (predicted == labels).sum()
  # validationAcc = correct / total
  print('-- EPOCH ' + str(i+1) + ' DONE.')
  print('Time elapsed: ' + str(time.time() - t))
  # print('Sampled accuracies from ' + str(sampleBatches) + ' batches.')
  # print('(sampled) Attacked Train Accuracy of CNN: ' + str(trainAcc * 100) + '%')
  # print('(sampled) Attacked Validation Accuracy of CNN: ' + str(validationAcc * 100) + '%')
  print('--')

  torch.save(model.state_dict(), 'out/robustClassifier_latest.pkl')
  print('Stored new robust classifier')
  print('--')



