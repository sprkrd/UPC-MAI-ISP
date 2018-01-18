import torch
import os
import torch.nn as nn
from torchvision.models import resnet18


def pretrained_res18(which=0, gpu=False):
    assert which in range(2)
    model = resnet18()
    model.fc = nn.Linear(512, 4)
    # path = os.path.join(os.path.dirname(__file__), "pretrained", "model224_cpu.pkl")
    if which == 0:
        filename = "res18_cpu.pkl"
    elif which == 1:
        filename = "res18_pgd_mild.pkl"
    elif which == 2:
        filename = "second_base.pkl"
    path = os.path.join(os.path.dirname(__file__), "pretrained", filename)
    state_dict = torch.load(path, map_location=lambda stg, _: stg)
    model.load_state_dict(state_dict)
    if gpu:
        model.cuda()
    model.eval()
    return model

