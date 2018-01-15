import torch
import os
import torch.nn as nn
from torchvision.models import resnet18


def pretrained_res18():
    model = resnet18()
    model.fc = nn.Linear(512, 4)
    # path = os.path.join(os.path.dirname(__file__), "pretrained", "model224_cpu.pkl")
    path = os.path.join(os.path.dirname(__file__), "pretrained", "res18_cpu.pkl")
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

