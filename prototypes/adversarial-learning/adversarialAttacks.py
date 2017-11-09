import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


##
# Run on GPU?
##
runGPU = False
if runGPU:
	cudnn.benchmark = True


##
# Hyper-parameters:
##

batch_size = 100
num_epochs_cnn = 5
learning_rate_cnn = 0.001
num_epochs_att = 5
learning_rate_att = 0.001
intensity = 0.1

##
# CNN model:
##
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out      

cnn = CNN()
if runGPU:
	cnn.cuda()



##
# Attacker model:
##
class Attacker(torch.nn.Module):
    def __init__(self):
        super(Attacker, self).__init__()
        ## encoder:
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        ## decoder:
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.Tanh())
    
    def forward(self, x):
        out = self.enc1(x)
        out = self.enc2(out)
        out = self.enc3(out)
        out = self.enc4(out)
        out = self.dec1(out)
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.dec4(out)
        out = out.clamp(min=-intensity, max=intensity)
        return out

attacker = Attacker()
if runGPU:
	attacker.cuda()

##
# Load Data:
##
train_dataset = dsets.MNIST(root='./data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data/',
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)



##
# Train CNN
##

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate_cnn)

# Train the Model
for epoch in range(num_epochs_cnn):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        if runGPU:
        	images.cuda()
        	labels.cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs_cnn, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    if runGPU:
        	images.cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the CNN1 on the 10000 test images: ' + str(100 * correct / total) + '%')

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn1.pkl')
print('CNN1 saved to cnn.pkl')
print('= ' * 20)


##
# Train Attacker
##
optimizer = torch.optim.Adam(attacker.parameters(), lr=learning_rate_att)
criterion = nn.CrossEntropyLoss()

# ! freeze cnn:
for param in cnn.parameters():
    param.requires_grad = False


for epoch in range(num_epochs_att):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        if runGPU:
        	images.cuda()
        	labels.cuda()
        # Forward
        attacks = attacker(images)
        inputs = images + attacks
        preds = cnn(inputs)
        
        # Backward + Optimize:
        optimizer.zero_grad()
        loss = -criterion(preds, labels) #+ torch.max(attacks[0][0])* 100
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs_att, i+1, len(train_dataset)//batch_size, loss.data[0]))
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        if runGPU:
        	images.cuda()
        attacks = attacker(images)
        outputs = cnn(images + attacks)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Test Accuracy of the CNN1 on the attacked 10000 test images: ' + str(100 * correct / total) + '%')

# Save the Trained Model
torch.save(attacker.state_dict(), 'attacker.pkl')
print('Attacker saved to attacker.pkl')
print('= ' * 20)



##
# Train CNN2
##
cnn2 = CNN()
print('Training CNN2')


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn2.parameters(), lr=learning_rate_cnn)

# Train the Model
for epoch in range(num_epochs_cnn):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        if runGPU:
            images.cuda()
            labels.cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn2(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs_cnn, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
cnn2.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    if runGPU:
            images.cuda()
    outputs = cnn2(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the CNN2 on the 10000 test images: ' + str(100 * correct / total) + '%')

cnn2.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    if runGPU:
        images.cuda()
    attacks = attacker(images)
    outputs = cnn2(images + attacks)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Test Accuracy of the CNN2 on the attacked 10000 test images: ' + str(100 * correct / total) + '%')

# Save the Trained Model
torch.save(cnn2.state_dict(), 'cnn2.pkl')
print('CNN saved to cnn2.pkl')
print('= ' * 20)