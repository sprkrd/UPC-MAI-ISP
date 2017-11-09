import os
from torch.utils.data import Dataset
from skimage import io, transform
import numpy as np



class EuroNotes(Dataset):
	""" EuroNotes dataset """

	def __init__(self, root_dir, transform=None):
		"""
		Args:
			root_dir (string): root directory where data is stored
			transform (callable): optional transform to be applied on a sample
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.images = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f)) and f.endswith(".jpg")]
		self.labels = [5, 10, 20, 50]

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = io.imread(os.path.join(self.root_dir, self.images[idx]))
		# TODO: delete next line:
		image = transform.resize(image, (3,256,256))
		label = self.labels.index(int(self.images[idx].split('_')[1]))
		sample = {'image': image, 'label': label}
		if self.transform:
			sample = self.transform(sample)
		return sample