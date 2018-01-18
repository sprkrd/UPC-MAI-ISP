# so it can be executed both with Python2 and Python3
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
import time

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.transform import resize

from .data.EuroNotes import EuroNotes
from .attackers.WhiteBoxAttacker import PGDAttack
from .attackers.BlackBoxAttacker import GANAttack, WhiteNoiseAttack
from .utils import means, stds
from .models.ResNet18 import pretrained_res18


class OutFeed:
    def __init__(self):
        self.last_msg = ""

    def print(self, msg, *args, **kwargs):
        kwargs["end"] = ""
        empty = not self.last_msg
        padding = len(self.last_msg) - len(msg)
        self.last_msg = msg
        if padding > 0:
            msg = msg + " "*padding
        print(msg if empty else "\r"+msg, *args, **kwargs)

    def newline(self, msg, *args, **kwargs):
        self.last_msg = ""
        print("\n"+msg, *args, **kwargs)


def evaluate(model, data_set, attacker=None, attack_mode='white', sampleBatches=None, prefix="",
             batch_size=25):
    if model.training:
        model.eval()
    loader = DataLoader(data_set, batch_size=batch_size,
                        shuffle=True, num_workers=2)
    sampleBatches = sampleBatches or len(loader)
    out = OutFeed()
    correct = 0
    total = 0
    for i_batch, data in enumerate(loader):
        if i_batch >= sampleBatches: 
            break
        images = data['image']
        labels = data['label']
        if runGPU:
            images = images.cuda()
            labels = labels.cuda()
        if attacker:
            if attack_mode == 'white':
                images = attacker.attack(model, images, labels, wrap=False)
            else:
                images = attacker.attack(images)
        images = Variable(images, volatile=True)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # predicted = predicted.type(torch.LongTensor)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        acc = 100 * correct / total
        msg = '[{}/{}] {}: Accuracy of CNN: {:.02f}%'.format(
                i_batch+1, sampleBatches, prefix, acc)
        out.print(msg)
    out.newline("{}: Final accuracy: {:.02f}%".format(prefix, acc))
    return acc


if __name__ == "__main__":
    runGPU = True
    if runGPU:
        cudnn.benchmark = True

    transformations = transforms.Compose([
        lambda img: resize(img, (224,224), preserve_range=True, mode="constant"),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])

    train_set = EuroNotes('data-augmentation/banknotes_augmented/train',
                          transform=transformations, resize=False)
    val_set = EuroNotes('data-augmentation/banknotes_augmented/val',
                          transform=transformations, resize=False)
    test_set = EuroNotes('data-augmentation/banknotes_augmented/test',
                        transform=transformations, resize=False)

    # which=0 basic model
    # which=1 trained against PGD for 5 epochs (only against mild attacks)
    # which=2 trained against PGD for 5 epochs (against strong attacks)
    cnn = pretrained_res18(which=2, gpu=runGPU)

    # attacker_mild = PGDAttack(k=5, epsilon=0.03) # mild attack
    # attacker_strong = PGDAttack(k=10, epsilon=0.05) # strong attack
    # attacker_gan = GANAttack(None, 0.2)
    # attacker_wn = WhiteNoiseAttack(intensity=1.0)


    # sampleBatches = 10
    # evaluate(cnn, train_set, sampleBatches=sampleBatches, prefix="Train set")
    # evaluate(cnn, val_set, sampleBatches=sampleBatches, prefix="Validation set")

    # evaluate(cnn, test_set, prefix="Test set (no attack)", attacker=None)
    # evaluate(cnn, test_set, prefix="Test set (mild attack)", attacker=attacker_mild)
    # evaluate(cnn, test_set, prefix="Test set (strong attack)", attacker=attacker_strong)
    # evaluate(cnn, test_set, prefix="Test set (GAN attack)", attacker=attacker_gan, attack_mode='black')
    evaluate(cnn, test_set, prefix="Test set (white noise attack)", attacker=attacker_wn, attack_mode='black')

