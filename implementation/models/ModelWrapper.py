import torch.nn

from torchvision import transforms
from torch.autograd import Variable

from skimage.transform import resize


class ModelWrapper:

    def __init__(self, model, means=None, std=None):
        self.model = model
        means = means or [0.14588552, 0.26887908, 0.14538361]
        std = std or [0.20122388, 0.2800698 , 0.20029236]
        self.tform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, std),
        ])
        self.softmax = torch.nn.Softmax(1)

    def __call__(self, img):
        img = self.tform(resize(img, (224, 224), mode="constant"))
        batch_shape = [1,] + list(img.size())
        batch = img.view(*batch_shape)
        x = Variable(batch, volatile=True)
        y = self.softmax(self.model(x))
        return y.data.tolist()[0]


if __name__ == "__main__":
    from skimage.io import imread
    from ResNet18 import pretrained_res18
    # quick test
    model = pretrained_res18()
    wrapper = ModelWrapper(model)
    img = imread("../../data-augmentation/banknotes_augmented_small/test/img_10_100_2.jpg")
    print(wrapper(img))

