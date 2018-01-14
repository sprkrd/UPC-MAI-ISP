import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from data.EuroNotes import EuroNotes

from models.BasicCNN import BasicCNN

import numpy as np

import time


##
# Run on GPU?
##
runGPU = True
if runGPU:
	cudnn.benchmark = True


# means and stds of the train set can be obtained with 
# print(train_set.getMeansAndStdPerChannel())
# this takes quite a while, so hard-coded here:
# means = np.array([ 0.14588552,  0.26887908,  0.14538361])
# stds = np.array([ 0.20122388,  0.2800698 ,  0.20029236])
means = np.array([ 0.34065133, 0.30230788, 0.27947797])
stds = np.array([ 0.28919015, 0.26877816, 0.25182973])




transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, stds)])

train_set = EuroNotes('data-augmentation/banknotes_augmented/train', transform=transformations, resize=False)
val_set = EuroNotes('data-augmentation/banknotes_augmented/val', transform=transformations, resize=False)


train_loader = DataLoader(train_set, batch_size=25, shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size=25, shuffle=True, num_workers=16)

cnn = torchvision.models.resnet18(pretrained=True)
# for param in cnn.parameters():
# 	param.requires_grad = False
cnn.fc = nn.Linear(512, 4)




if runGPU:
	cnn.cuda()
else:
	cnn.cpu()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters())


epochs = 100

bestValidationAcc = 0.0

for i in range(0, epochs):
	cnn.train()

	losses = 0
	lossSum = 0.0
	t = time.time()
	for i_batch, sample_batched in enumerate(train_loader):
		
		images = Variable(sample_batched['image'])
		labels = Variable(sample_batched['label'])
		if runGPU:
			images = images.cuda()
			labels = labels.cuda()

		optimizer.zero_grad()

		outputs = cnn(images)

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
	
	cnn.eval()
	
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
			outputs = cnn(images)
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
		outputs = cnn(images)
		_, predicted = torch.max(outputs.data, 1)
		predicted = predicted.type(torch.LongTensor)
		total += labels.size(0)
		correct += (predicted == labels).sum()
	validationAcc = correct / total
	print('-- EPOCH ' + str(i+1) + ' DONE.')
	print('Time elapsed: ' + str(time.time() - t))
	print('Sampled accuracies from ' + str(sampleBatches) + ' batches.')
	print('(sampled) Train Accuracy of CNN: ' + str(trainAcc * 100) + '%')
	print('(sampled) Validation Accuracy of CNN: ' + str(validationAcc * 100) + '%')
	print('--')

	if validationAcc > bestValidationAcc:
		bestValidationAcc = validationAcc
		torch.save(cnn.state_dict(), 'out/baseClassifier_best.pkl')
		print('Stored new best base classifier')
		print('--')

