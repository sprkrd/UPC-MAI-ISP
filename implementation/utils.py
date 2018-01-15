import numpy as np
import torch

from torch.autograd import Variable
from torchvision import transforms
from skimage.transform import resize


means = np.array([0.34065133, 0.30230788, 0.27947797])
stds = np.array([0.28919015, 0.26877816, 0.25182973])


def retrieve_image(variable, means=means, stds=stds):
    img = variable.data.numpy()[0,:]
    img = img.swapaxes(0, 1).swapaxes(1, 2)
    img = np.array(np.round(255*(img*stds + means)), dtype=np.uint8)
    return img


def center_crop(img):
    h, w = img.shape[:2]
    side = min(h, w)
    slack_v = h - side
    slack_h = w - side
    slack_top = slack_v // 2
    slack_bottom = slack_top + slack_v%2
    slack_left = slack_h // 2
    slack_right = slack_left + slack_h%2
    return img[slack_top:h-slack_bottom,slack_left:w-slack_right,:].copy()


def do_batchx(img):
    batch_shape = [1,] + list(img.size())
    x = Variable(img.view(*batch_shape), volatile=True)
    return x


def do_batchy(label):
    return Variable(torch.Tensor([label]), volatile=True)



tform1 = transforms.Compose([
    center_crop,
    lambda img: resize(img, (224, 224), preserve_range=True, mode="constant"),
])


tform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
    do_batchx,
])

