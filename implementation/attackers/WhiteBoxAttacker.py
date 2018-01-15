import torch
import torch.nn
import torch.nn.functional as F

from torch.autograd import Variable
from .Attacker import Attacker

from ..utils import do_batchy, tform1, tform2, retrieve_image, means, stds


class WhiteBoxAttacker(Attacker):

    def __init__(self, attack_shape):
        super(WhiteBoxAttacker, self).__init__(attack_shape)
        
    def attack(self, model, input_data, target):
        raise NotImplementedError
        
    def feedback(self, last_corrects):
        raise NotImplementedError


class PGDAttack(WhiteBoxAttacker):

    def __init__(self, epsilon=0.03, a=0.01, k=40, attack_shape=None):
        super(PGDAttack, self).__init__(attack_shape)
        self.epsilon = epsilon
        self.a = a
        self.k = k

    def attack(self, model, x_nat, y):
        """ Takes an input batch and adds a small perturbation """
        min_values = -means/stds
        max_values = (1-means)/stds
        epsilon = self.epsilon # Maximum strength of the perturbation
        a = self.a # maximum change to the image
        k = self.k
        z = torch.rand(x_nat.size())
        x = Variable(x_nat + 2*epsilon*(torch.rand(x_nat.size())-0.5), requires_grad=True)
        y = Variable(y)
        for _ in range(k):
            y_ = model(x)
            loss = F.cross_entropy(y_, y)
            loss.backward()
            x = x + a*torch.sign(x.grad)
            x = torch.min(torch.max(x.data, x_nat-epsilon), x_nat+epsilon)
            for ch in range(3):
                x[:,ch,:,:] = torch.clamp(x[:,ch,:,:], min_values[ch], max_values[ch])
            x = Variable(x, requires_grad=True)
        return x.detach(), y

    def __call__(self, model, img, return_img=False):
        s = torch.nn.Softmax(1)
        x = tform2(tform1(img))
        y = model(x)
        _, label = torch.max(y.data, 1)
        x_pert, _ = self.attack(model, x.data, label)
        y = s(model(x_pert))
        y = y.data.tolist()[0]
        if return_img:
            img_pert = retrieve_image(x_pert)
            return img_pert, y
        return y


if __name__ == "__main__":
    from ..models.ResNet18 import pretrained_res18
    from ..models.ModelWrapper import ModelWrapper
    from skimage.io import imread, imshow
    import matplotlib.pyplot as plt
    import numpy as np
    # quick test
    model = pretrained_res18()
    wrapper = ModelWrapper(model)
    attack = PGDAttack(epsilon=0.03, k=5)
    # img = imread("../../data-augmentation/banknotes_augmented/val/img_10_76_98.jpg")
    # img = imread("data-augmentation/banknotes_augmented/test/img_5_90_10.jpg")
    # img = imread("../../data-augmentation/banknotes_augmented/test/img_20_133_100.jpg")
    # img = imread("../../data-augmentation/banknotes_augmented/val/img_50_71_2.jpg")
    img = imread("/home/sprkrd/Pictures/banknotes/10/IMG_20171009_192337.jpg")
    img, p1 = wrapper(img, True)
    img_pert, p2 = attack(model, img, True)
    print(p1)
    print(p2)
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.title("Original image")
    plt.subplot(2,2,2)
    plt.imshow(img_pert)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.title("Perturbed image (PGD)")
    plt.subplot(2,2,3)
    plt.title("Probability distribution")
    plt.bar([1.5,2.5,3.5,4.5], p1, tick_label=["5", "10", "20", "50"], log=True)
    plt.xlabel("Banknote")
    plt.subplot(2,2,4)
    plt.title("Probability distribution")
    plt.bar([1.5,2.5,3.5,4.5], p2, tick_label=["5", "10", "20", "50"], log=True)
    plt.xlabel("Banknote")
    plt.tight_layout()
    plt.show()
    diff = np.abs(img_pert.astype(np.float) - img)
    diff = np.mean(diff, 2)
    min_ = np.min(diff)
    max_ = np.max(diff)
    diff = (diff-min_)/(max_-min_)
    plt.imshow(diff, cmap=plt.get_cmap("hot"))
    plt.title("Normalized differences (maximum diff: {:.00f})".format(max_))
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.colorbar()
    plt.tight_layout()
    plt.show()





