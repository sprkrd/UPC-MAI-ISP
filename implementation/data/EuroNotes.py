import os
from torch.utils.data import Dataset
from skimage import io, transform
from skimage.util.dtype import convert
import numpy as np

class EuroNotes(Dataset):
	""" EuroNotes dataset """

	def __init__(self, root_dir, transform=None, resize=False):
		"""
		Args:
			root_dir (string): root directory where data is stored
			transform (callable): optional transform to be applied on a sample
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.resize = resize
		self.images = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f)) and f.endswith(".jpg")]
		self.labels = [5, 10, 20, 50]

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = io.imread(os.path.join(self.root_dir, self.images[idx]))
		label = self.labels.index(int(self.images[idx].split('_')[1]))
		sample = {'image': image, 'label': label}
		if self.transform:
			if self.resize:
				# resize image somewhere between 100 and 120%:
				newSize = int(np.random.uniform(1.0, 1.05) * 256)
				sample['image'] = transform.resize(sample['image'], (258, 258))
				sample['image'] = convert(sample['image'], np.uint8)
			sample['image'] = self.transform(sample['image'])
		return sample

	def getMeansAndStdPerChannel(self):
		imagesMeans = np.zeros((3, self.__len__()))
		imagesStds = np.zeros((3, self.__len__()))
		for idx in tqdm(range(0, self.__len__())):
			image = self.__getitem__(idx)['image']
			imageMeans = np.mean(image.numpy(), axis=(1, 2))
			imageStds = np.std(image.numpy(), axis=(1, 2))
			imagesMeans[:, idx] = imageMeans
			imagesStds[:, idx] = imageStds

		means = np.mean(imagesMeans, axis = 1)
		stds = np.mean(imagesStds, axis = 1)

		print(means.shape)
		print(stds.shape)

		return means, stds
