import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data.EuroNotes import EuroNotes
from data.transforms import to_tensor

from models.BasicCNN import BasicCNN

import time

train_set = EuroNotes('data/small/train', transform=to_tensor)
val_set = EuroNotes('data/small/val', transform=to_tensor)


train_loader = DataLoader(train_set, batch_size=40, shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size=40, shuffle=True, num_workers=16)

cnn = BasicCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters())


epochs = 10

for i in range(0, epochs):
	cnn.train()
	for i_batch, sample_batched in enumerate(train_loader):
		
		images = Variable(sample_batched['image'])
		labels = Variable(sample_batched['label'])

		optimizer.zero_grad()

		outputs = cnn(images)

		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		print(loss.data[0])
	
	cnn.eval()
	correct = 0.0
	total = 0.0
	for data in val_loader:
		images = Variable(data['image'])
		labels = data['label'].type(torch.LongTensor)
		outputs = cnn(images)
		_, predicted = torch.max(outputs.data, 1)
		predicted = predicted.type(torch.LongTensor)
		total += labels.size(0)
		correct += (predicted == labels).sum()
	print('-- Test Accuracy of CNN on test images: ' + str(100 * correct / total) + '%')



