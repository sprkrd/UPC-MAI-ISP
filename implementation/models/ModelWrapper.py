import numpy as np
import torch.nn

from ..utils import tform1, tform2


class ModelWrapper:

    def __init__(self, model):
        self.model = model
        self.softmax = torch.nn.Softmax(1)

    def __call__(self, img, return_img=False):
        img = tform1(img)
        x = tform2(img)
        y = self.softmax(self.model(x))
        y = y.data.tolist()[0]
        return (np.array(img, dtype=np.uint8), y) if return_img else y


if __name__ == "__main__":
    from .ResNet18 import pretrained_res18
    from skimage.io import imread, imshow
    import matplotlib.pyplot as plt
    # quick test
    model = pretrained_res18()
    wrapper = ModelWrapper(model)
    # img = imread("../../data-augmentation/banknotes_augmented/val/img_10_76_98.jpg")
    # img = imread("../../data-augmentation/banknotes_augmented/val/img_5_62_23.jpg")
    # img = imread("../../data-augmentation/banknotes_augmented/test/img_20_133_100.jpg")
    # img = imread("../../data-augmentation/banknotes_augmented/val/img_50_71_2.jpg")
    img = imread("/home/sprkrd/Pictures/banknotes/10/IMG_20171009_192337.jpg")
    # plt.imshow(img)
    # plt.show()
    img, p = wrapper(img, True)
    plt.imshow(img)
    print(p)
    plt.show()

