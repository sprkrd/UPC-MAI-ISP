import numpy as np
import torch

from torch.autograd import Variable

from .Attacker import Attacker
from .WhiteBoxAttacker import PGDAttack

from ..models.PixelLevelTransferN import PixelLevelTransferN

from ..utils import clamp_to_valid_img, tform1, tform2, retrieve_image


class BlackBoxAttacker(Attacker):
  
    def __init__(self, attack_shape):
        Attacker.__init__(self, attack_shape)
    
    def attack(self, input_data):
        raise NotImplementedError
    
    def feedback(self, last_corrects):
        raise NotImplementedError

    def __call__(self, model, img, return_img=False):
        """
        Performs an undirected attack against the given model and returns the
        classifier's output PDF and (optionally) the perturbed image
        """
        s = torch.nn.Softmax(1)
        x = tform2(tform1(img))
        x_pert = Variable(self.attack(x.data), volatile=True)
        y = s(model(x_pert))
        y = y.data.tolist()[0]
        if return_img:
            img_pert = retrieve_image(x_pert)
            return img_pert, y
        return y


class PGDAttackBB(BlackBoxAttacker):
    def __init__(self, model, attack_shape=None, epsilon=0.03, a=0.01, k=40):
        super(PGDAttackBB, self).__init__(attack_shape)
        self.model = model
        self.attacker = PGDAttack(epsilon=epsilon, a=a, k=k)

    def attack(self, input_data):
        # obtain label predicted by attacker's model
        y = self.model(Variable(input_data, volatile=True))
        _, label = torch.max(y.data, 1)
        x_pert = self.attacker.attack(self.model, input_data, label)
        return x_pert


class GANAttack(BlackBoxAttacker):

    def __init__(self, attack_shape=None, intensity=0.2):
        BlackBoxAttacker.__init__(self, attack_shape)

        attacker = PixelLevelTransferN(in_channels=3, out_channels=3, intensity=intensity)
        state_dict = torch.load('implementation/models/pretrained/gan_attacker.pkl')
        attacker.load_state_dict(state_dict)
        self.attacker = attacker


    def attack(self, input_data):

        images = Variable(input_data, volatile=True)
        images = (images + self.attacker(images)).data
        images = clamp_to_valid_img(images)
        return images


class WhiteNoiseAttack(BlackBoxAttacker):

    def __init__(self, attack_shape=None, intensity=0.2):
        BlackBoxAttacker.__init__(self, attack_shape)
        self.intensity = intensity

    def attack(self, input_data):
        noise = np.random.uniform(low=-self.intensity, high=self.intensity, size=np.array(tuple(input_data.shape)))
        noise = torch.from_numpy(noise).float()
        images = input_data + noise
        images = clamp_to_valid_img(images)
        return images

if __name__ == "__main__":
    # gan_attacker = GANAttack(None, None)
    from ..models.ResNet18 import pretrained_res18
    from ..models.ModelWrapper import ModelWrapper
    from skimage.io import imread, imshow
    from ..utils import heat_map
    import matplotlib.pyplot as plt
    import numpy as np
    # quick test
    model = pretrained_res18()
    model_att = pretrained_res18(which=3)
    wrapper = ModelWrapper(model)
    attack = PGDAttackBB(model, epsilon=0.03, k=5)
    # img = imread("../../data-augmentation/banknotes_augmented/val/img_10_76_98.jpg")
    # img = imread("data-augmentation/banknotes_augmented/test/img_5_90_10.jpg")
    # img = imread("../../data-augmentation/banknotes_augmented/test/img_20_133_100.jpg")
    # img = imread("../../data-augmentation/banknotes_augmented/val/img_50_71_2.jpg")
    img = imread("data-augmentation/banknotes_augmented_small/test/img_10_100_1.jpg")
    img, p1 = wrapper(img, True)
    img_pert, p2 = attack(model_att, img, True)
    # print(p1)
    # print(p2)
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.title("Original image")
    plt.subplot(2,2,2)
    plt.imshow(img_pert)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.title("Perturbed image (undirected PGD)")
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
    plt.close()
    # diff = np.abs(img_pert.astype(np.float) - img)
    # diff = np.mean(diff, 2)
    # min_ = np.min(diff)
    # max_ = np.max(diff)
    # diff = (diff-min_)/(max_-min_)
    # plt.imshow(diff, cmap=plt.get_cmap("hot"))
    # plt.title("Normalized differences (maximum diff: {:.00f})".format(max_))
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    heatmap = heat_map(img, img_pert)
    plt.imshow(heatmap)
    plt.show()
    
