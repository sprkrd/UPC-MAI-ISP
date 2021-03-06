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
from models.PixelLevelTransferN import PixelLevelTransferN


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
model.eval()
# freeze classifier:
for param in model.parameters():
    param.requires_grad = False

attacker = PixelLevelTransferN(intensity=0.2, in_channels=3, out_channels=3)

if runGPU:
    model.cuda()
    attacker.cuda()
else:
    model.cpu()
    attacker.cpu()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(attacker.parameters())


epochs = 100

bestValidationAcc = 1.0

for i in range(0, epochs):
    attacker.train()

    losses = 0
    lossSum = 0.0
    t = time.time()
    for i_batch, sample_batched in enumerate(train_loader):

        print(i_batch)

        images = Variable(sample_batched['image'])
        labels = Variable(sample_batched['label'])
        if runGPU:
            images = images.cuda()
            labels = labels.cuda()
        # Forward
        attacks = attacker(images)
        inputs = images + attacks
        preds = model(inputs)
        
        # Backward + Optimize:
        optimizer.zero_grad()
        loss = -criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        losses += 1
        lossSum += loss.data[0]
        batchesPerReport = 100
        if i_batch % batchesPerReport == 0:
            print('> batch ' + str(i_batch) + ', current average loss: ' + str(lossSum / losses))
            losses = 0
            lossSum = 0

    attacker.eval()
    
    sampleBatches = 100

    correct = 0.0   
    total = 0.0
    for i_batch, data in enumerate(train_loader):
        if i_batch > sampleBatches:
            break
        images = Variable(data['image'], volatile=True)
        labels = data['label'].type(torch.LongTensor)
        if runGPU:
            images = images.cuda()
        attacks = attacker(images)
        outputs = model(images + attacks)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.type(torch.LongTensor)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    trainAcc = correct / total

    correct = 0.0
    total = 0.0
    for i_batch, data in enumerate(val_loader):
        if i_batch > sampleBatches:
            break
        images = Variable(data['image'], volatile=True)
        labels = data['label'].type(torch.LongTensor)
        if runGPU:
            images = images.cuda()
        attacks = attacker(images)
        outputs = model(images + attacks)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.type(torch.LongTensor)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    validationAcc = correct / total
    print('-- EPOCH ' + str(i+1) + ' DONE.')
    print('Time elapsed: ' + str(time.time() - t))
    print('Sampled accuracies from ' + str(sampleBatches) + ' batches.')
    print('(sampled) Attacked Train Accuracy of CNN: ' + str(trainAcc * 100) + '%')
    print('(sampled) Attacke Validation Accuracy of CNN: ' + str(validationAcc * 100) + '%')
    print('--')

    if validationAcc < bestValidationAcc:
        bestValidationAcc = validationAcc
        torch.save(attacker.state_dict(), 'out/bestAttacker_best.pkl')
        print('Stored new best base classifier')
        print('--')


