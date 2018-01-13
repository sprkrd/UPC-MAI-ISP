import torch
import torch.nn as nn
from torchvision.models import resnet18



def pretrained_res18():
	model = resnet18()
	model.fc = nn.Linear(512, 4)
	model.load_state_dict(torch.load('./models/pretrained/res18.pkl'))
	return model