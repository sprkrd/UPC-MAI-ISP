from mnist import MNIST
from toTensor import to_tensor as toTensor
import argparse

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time 

##
# Experiment structure:
##
#
# A: (search for EXPERIMENT PART A)
# train CNN1 and CNN2 on real data, 20 epochs
# train Attacker based on CNN1 loss for 20 epochs
# evaluate Attacker performance on CNN1 and CNN2 after each minibatch
#
#
# B: (search for EXPERIMENT PART B)
# train CNN1 and CNN2 on real data, 5 epochs
# train Attacker and CNN1 based on CNN1 loss alternating for 20 epochs
# train CNN2 on real data for 20 more epochs
# evaluate Attacker performance on CNN1 and CNN2 after each minibatch 
# (starting evaluation in the 6th epoch of CNN1 and CNN2)


##
# Run on GPU?
##
runGPU = True
if runGPU:
    cudnn.benchmark = True


##
# Hyper-parameters:
##
intensities = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
batch_size = 100

##
# Experiment settings:
## 
expA_cnn1_epochs = 20
expA_cnn2_epochs = 20
expA_att_epochs = 20
expB_headstart_epochs = 5
expB_adversarial_epochs = 20


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
        # out = out.clamp(min=-intensity, max=intensity)
	out = out * intensity
        return out

##
# Load Data:
##
train_dataset = MNIST(root='data/', train=True, transform=toTensor, download=True)
test_dataset = MNIST(root='data/', train=False, transform=toTensor, download=True) 


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


##
# RUN EXPERIMENTS FOR EACH INTENSITY:
##
for intensity in intensities:

    expName = 'in_' + str(intensity) + '_'

    print('# ' * 21)
    print('RUN EXPERIMENTS A & B FOR INTENSITY = ' + str(intensity))
    print('# ' * 21)


    ##
    # EXPERIMENT PART A:
    ## 
    print('= ' * 20)
    print('# Experiment part A')
    print('= ' * 20)

    ##
    # Experiment A - CNN1:
    ##

    print('## Training CNN1 for ' + str(expA_cnn1_epochs) + ' epochs on real data')

    t = time.time()

    cnn1 = CNN()
    if runGPU:
        cnn1.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn1.parameters(), lr=0.001)

    # Losses after each minibatch:
    losses = []
    # Accuracies on test set after each epoch:
    accuracies = []

    # Train the Model
    for epoch in range(expA_cnn1_epochs):
        cnn1.train()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            if runGPU:
                images = images.cuda()
                labels = labels.cuda()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn1(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.data[0])
            
            if (i+1) % 100 == 0:
                print ('-- Epoch [%d/%d], Iter [%d/%d] Loss: %.5f' 
                       %(epoch+1, expA_cnn1_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
        
        # estimate test accuracy:
        cnn1.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = Variable(images)
            if runGPU:
                    images = images.cuda()
            outputs = cnn1(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.LongTensor)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('-- Test Accuracy of CNN1 on test images: ' + str(100 * correct / total) + '%')
        accuracies.append(correct/total)

    # Save losses and accuracies for later analysis:
    with open("out/" + expName + "A_CNN1_losses.data", "w") as output:
        output.write(str(losses))
    with open("out/" + expName + "A_CNN1_accuracies.data", "w") as output:
        output.write(str(accuracies))

    torch.save(cnn1.state_dict(), "out/" + expName + "A_cnn1.pkl")
    print("CNN2 saved to out/" + expName + "A_cnn1.pkl")

    print('Total training time CNN1: ' + str(time.time() - t))
    print('= ' * 20)

    ##
    # Experiment A - CNN2:
    ##

    print('## Training CNN2 for ' + str(expA_cnn2_epochs) + ' epochs on real data')

    t = time.time()

    cnn2 = CNN()
    if runGPU:
        cnn2.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn2.parameters(), lr=0.001)

    # Losses after each minibatch:
    losses = []
    # Accuracies on test set after each epoch:
    accuracies = []

    # Train the Model
    for epoch in range(expA_cnn2_epochs):
        cnn2.train()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            if runGPU:
                images = images.cuda()
                labels = labels.cuda()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn2(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.data[0])
            
            if (i+1) % 100 == 0:
                print ('-- Epoch [%d/%d], Iter [%d/%d] Loss: %.5f' 
                       %(epoch+1, expA_cnn2_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
        
        # estimate test accuracy:
        cnn2.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = Variable(images)
            if runGPU:
                    images = images.cuda()
            outputs = cnn2(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.LongTensor)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('-- Test Accuracy of CNN2 on test images: ' + str(100 * correct / total) + '%')
        accuracies.append(correct/total)

    # Save losses and accuracies for later analysis:
    with open("out/" + expName + "A_CNN2_losses.data", "w") as output:
        output.write(str(losses))
    with open("out/" + expName + "A_CNN2_accuracies.data", "w") as output:
        output.write(str(accuracies))

    torch.save(cnn2.state_dict(), "out/" + expName + "A_cnn2.pkl")
    print("CNN2 saved to out/" + expName + "A_cnn2.pkl")

    print('Total training time CNN2: ' + str(time.time() - t))
    print('= ' * 20)

    ##
    # Experiment A - Attacker:
    ##

    print('## Training Attacker for ' + str(expA_att_epochs) + ' epochs on loss of frozen CNN1')

    t = time.time()

    attacker = Attacker()
    if runGPU:
        attacker.cuda()


    # Loss and optimizer:
    optimizer = torch.optim.Adam(attacker.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Freeze the CNNs so they don't get changed during training of the attacker:
    for param in cnn1.parameters():
        param.requires_grad = False
    for param in cnn2.parameters():
        param.requires_grad = False

    # Losses after each minibatch:
    losses_cnn1 = []
    losses_cnn2 = []
    # Accuracies on test set after each epoch:
    accuracies_cnn1 = []
    accuracies_cnn2 = []


    # Train the model:
    for epoch in range(expA_att_epochs):
        attacker.train()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            if runGPU:
                images = images.cuda()
                labels = labels.cuda()
            # Forward
            attacks = attacker(images)
            inputs = images + attacks
            preds = cnn1(inputs)
            
            # Backward + Optimize:
            optimizer.zero_grad()
            loss = -criterion(preds, labels)
            loss.backward()
            optimizer.step()

            losses_cnn1.append(loss.data[0])
            loss_cnn2 = -criterion(cnn2(inputs), labels)
            losses_cnn2.append(loss_cnn2.data[0])
             
            if (i+1) % 100 == 0:
                print ('-- Epoch [%d/%d], Iter [%d/%d] Loss CNN1: %.5f, Loss CNN2: %.5f' 
                       %(epoch+1, expA_att_epochs, i+1, len(train_dataset)//batch_size, loss.data[0], loss_cnn2.data[0]))

        # estimate test accuracy:
        attacker.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = Variable(images)
            if runGPU:
                    images = images.cuda()
            attacks = attacker(images)
            outputs = cnn1(images + attacks)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.LongTensor)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('-- Test Accuracy of CNN1 on attacked test images: ' + str(100 * correct / total) + '%')
        accuracies_cnn1.append(correct/total)
        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = Variable(images)
            if runGPU:
                    images = images.cuda()
            attacks = attacker(images)
            outputs = cnn2(images + attacks)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.LongTensor)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('-- Test Accuracy of CNN2 on attacked test images: ' + str(100 * correct / total) + '%')
        accuracies_cnn2.append(correct/total)


    # Save losses and accuracies for later analysis:
    with open("out/" + expName + "A_ATT_lossesCNN1.data", "w") as output:
        output.write(str(losses_cnn1))
    with open("out/" + expName + "A_ATT_lossesCNN2.data", "w") as output:
        output.write(str(losses_cnn2))    
    with open("out/" + expName + "A_ATT_accuraciesCNN1.data", "w") as output:
        output.write(str(accuracies_cnn1))
    with open("out/" + expName + "A_ATT_accuraciesCNN2.data", "w") as output:
        output.write(str(accuracies_cnn2))

    torch.save(attacker.state_dict(), "out/" + expName + "A_attacker.pkl")
    print("Attacker saved to out/" + expName + "A_attacker.pkl")

    print('Total training time attacker: ' + str(time.time() - t))
    print('= ' * 20)

    ##
    # EXPERIMENT PART B:
    ## 
    print('= ' * 20)
    print('# Experiment part B')
    print('= ' * 20)

    ##
    # Experiment B - CNN1:
    ##

    print('## Training CNN1 for ' + str(expB_headstart_epochs) + ' epochs on real data')

    t = time.time()

    cnn1 = CNN()
    if runGPU:
        cnn1.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn1.parameters(), lr=0.001)

    # Losses after each minibatch:
    losses = []
    # Accuracies on test set after each epoch:
    accuracies = []

    # Train the Model
    for epoch in range(expB_headstart_epochs):
        cnn1.train()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            if runGPU:
                images = images.cuda()
                labels = labels.cuda()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn1(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.data[0])
            
            if (i+1) % 100 == 0:
                print ('-- Epoch [%d/%d], Iter [%d/%d] Loss: %.5f' 
                       %(epoch+1, expA_cnn1_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
        
        # estimate test accuracy:
        cnn1.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = Variable(images)
            if runGPU:
                    images = images.cuda()
            outputs = cnn1(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.LongTensor)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('-- Test Accuracy of CNN1 on test images: ' + str(100 * correct / total) + '%')
        accuracies.append(correct/total)

    # Save losses and accuracies for later analysis:
    with open("out/" + expName + "B_CNN1_losses.data", "w") as output:
        output.write(str(losses))
    with open("out/" + expName + "B_CNN1_accuracies.data", "w") as output:
        output.write(str(accuracies))

    torch.save(cnn1.state_dict(), "out/" + expName + "B_cnn1.pkl")
    print("CNN2 saved to out/" + expName + "B_cnn1.pkl")

    print('Total training time CNN1: ' + str(time.time() - t))
    print('= ' * 20)

    ##
    # Experiment A - CNN2:
    ##

    print('## Training CNN2 for ' + str(expB_headstart_epochs) + ' epochs on real data')

    t = time.time()

    cnn2 = CNN()
    if runGPU:
        cnn2.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn2.parameters(), lr=0.001)

    # Losses after each minibatch:
    losses = []
    # Accuracies on test set after each epoch:
    accuracies = []

    # Train the Model
    for epoch in range(expB_headstart_epochs):
        cnn2.train()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            if runGPU:
                images = images.cuda()
                labels = labels.cuda()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn2(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.data[0])
            
            if (i+1) % 100 == 0:
                print ('-- Epoch [%d/%d], Iter [%d/%d] Loss: %.5f' 
                       %(epoch+1, expA_cnn2_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
        
        # estimate test accuracy:
        cnn2.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = Variable(images)
            if runGPU:
                    images = images.cuda()
            outputs = cnn2(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.LongTensor)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('-- Test Accuracy of CNN2 on test images: ' + str(100 * correct / total) + '%')
        accuracies.append(correct/total)

    # Save losses and accuracies for later analysis:
    with open("out/" + expName + "B_CNN2_losses.data", "w") as output:
        output.write(str(losses))
    with open("out/" + expName + "B_CNN2_accuracies.data", "w") as output:
        output.write(str(accuracies))

    torch.save(cnn2.state_dict(), "out/" + expName + "B_cnn2.pkl")
    print("CNN2 saved to out/" + expName + "B_cnn2.pkl")

    print('Total training time CNN2: ' + str(time.time() - t))
    print('= ' * 20)

    ##
    # Experiment A - Adversarial Training:
    ##

    print('## Adversarial training between CNN1 and Attacker for ' + str(expB_adversarial_epochs) + ' epochs')
    print('## Continuing training of CNN2 for ' + str(expB_adversarial_epochs) + ' epochs on real data')

    t = time.time()

    attacker = Attacker()
    if runGPU:
        attacker.cuda()


    # Loss and optimizer:
    optimizerAtt = torch.optim.Adam(attacker.parameters(), lr=0.001)
    optimizerCNN1 = torch.optim.Adam(cnn1.parameters(), lr=0.001)
    optimizerCNN2 = torch.optim.Adam(cnn2.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Freeze the CNNs so they don't get changed during training of the attacker:
    for param in cnn1.parameters():
        param.requires_grad = False
    for param in cnn2.parameters():
        param.requires_grad = False

    # Losses after each minibatch:
    losses_cnn1 = []
    losses_cnn2_att = []
    losses_cnn2_real = []
    # Accuracies on test set after each epoch:
    accuracies_cnn1_att = []
    accuracies_cnn1_real = []
    accuracies_cnn2_att = []
    accuracies_cnn2_real = []


    # Train the model:
    for epoch in range(expB_adversarial_epochs):
        attacker.train()
        cnn1.train()
        cnn2.train()
        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images)
            labels = Variable(labels)
            if runGPU:
                images = images.cuda()
                labels = labels.cuda()


            # Optimize Attacker:
            for param in attacker.parameters():
                param.requires_grad = True
            for param in cnn1.parameters():
                param.requires_grad = False
            for param in cnn2.parameters():
                param.requires_grad = False
            attacks = attacker(images)
            inputs = images + attacks
            pred_cnn1_att = cnn1(inputs)
            optimizerAtt.zero_grad()
            lossAtt = -criterion(pred_cnn1_att, labels)
            lossAtt.backward()
            optimizerAtt.step()

            # Optimize CNN1:
            for param in attacker.parameters():
                param.requires_grad = False
            for param in cnn1.parameters():
                param.requires_grad = True
            for param in cnn2.parameters():
                param.requires_grad = False
            attacks = attacker(images)
            inputs = images + attacks
            pred_cnn1_att = cnn1(inputs)
            optimizerCNN1.zero_grad()
            lossCNN1 = criterion(pred_cnn1_att, labels)
            lossCNN1.backward()
            optimizerCNN1.step()


            # Optimize CNN2:
            for param in attacker.parameters():
                param.requires_grad = False
            for param in cnn1.parameters():
                param.requires_grad = False
            for param in cnn2.parameters():
                param.requires_grad = True
            pred_cnn2_real = cnn2(images)
            optimizerCNN2.zero_grad()
            lossCNN2_real = criterion(pred_cnn2_real, labels)
            lossCNN2_real.backward()
            optimizerCNN2.step()
            attacks = attacker(images)
            inputs = images + attacks
            pred_cnn2_att = cnn2(inputs)
            lossCNN2_att = criterion(pred_cnn2_att, labels)

            losses_cnn1.append(lossCNN1.data[0])
            losses_cnn2_att.append(lossCNN2_att.data[0])
            losses_cnn2_real.append(lossCNN2_real.data[0])

     
            if (i+1) % 100 == 0:
                print ('-- Epoch [%d/%d], Iter [%d/%d] Loss CNN1: %.5f, Loss CNN2 (real/attacked): %.5f / %.5f' 
                       %(epoch+1, expA_att_epochs, i+1, len(train_dataset)//batch_size, lossCNN1.data[0], lossCNN2_real.data[0], lossCNN2_att.data[0]))

        # estimate test accuracies:
        attacker.eval()  # Change models to 'eval' mode (BN uses moving mean/var).
        cnn1.eval()
        cnn2.eval()
        
        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = Variable(images)
            if runGPU:
                    images = images.cuda()
            attacks = attacker(images)
            outputs = cnn1(images + attacks)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.LongTensor)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('-- Test Accuracy of CNN1 on attacked test images: ' + str(100 * correct / total) + '%')
        accuracies_cnn1_att.append(correct/total)
        
        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = Variable(images)
            if runGPU:
                    images = images.cuda()
            attacks = attacker(images)
            outputs = cnn1(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.LongTensor)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('-- Test Accuracy of CNN1 on real test images: ' + str(100 * correct / total) + '%')
        accuracies_cnn1_real.append(correct/total)
        
        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = Variable(images)
            if runGPU:
                    images = images.cuda()
            attacks = attacker(images)
            outputs = cnn2(images + attacks)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.LongTensor)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('-- Test Accuracy of CNN2 on attacked test images: ' + str(100 * correct / total) + '%')
        accuracies_cnn2_att.append(correct/total)
        
        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = Variable(images)
            if runGPU:
                    images = images.cuda()
            attacks = attacker(images)
            outputs = cnn2(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.LongTensor)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('-- Test Accuracy of CNN2 on real test images: ' + str(100 * correct / total) + '%')
        accuracies_cnn2_real.append(correct/total)


    # Save losses and accuracies for later analysis:
    with open("out/" + expName + "B_ATT_lossesCNN1.data", "w") as output:
        output.write(str(losses_cnn1))
    with open("out/" + expName + "B_ATT_lossesCNN2real.data", "w") as output:
        output.write(str(losses_cnn2_real)) 
    with open("out/" + expName + "B_ATT_lossesCNN2att.data", "w") as output:
        output.write(str(losses_cnn2_att))    
    with open("out/" + expName + "B_ATT_accuraciesCNN1.data", "w") as output:
        output.write(str(accuracies_cnn1))
    with open("out/" + expName + "B_ATT_accuraciesCNN2real.data", "w") as output:
        output.write(str(accuracies_cnn2_real))
    with open("out/" + expName + "B_ATT_accuraciesCNN2att.data", "w") as output:
        output.write(str(accuracies_cnn2_att))

    torch.save(attacker.state_dict(), "out/" + expName + "B_attacker.pkl")
    print("Attacker saved to out/" + expName + "B_attacker.pkl")

    print('Total training time attacker: ' + str(time.time() - t))
    print('= ' * 20)


