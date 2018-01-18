import numpy as np

from .Attacker import Attacker

from ..models.PixelLevelTransferN import PixelLevelTransferN

from ..utils import clamp_to_valid_img

import torch
from torch.autograd import Variable

class BlackBoxAttacker(Attacker):
  
  def __init__(self, attack_shape):
    Attacker.__init__(self, attack_shape)
    
  def attack(self, input_data):
    raise NotImplementedError
    
  def feedback(self, last_corrects):
    raise NotImplementedError



class GANAttack(BlackBoxAttacker):

    def __init__(self, attack_shape, intensity):
        BlackBoxAttacker.__init__(self, attack_shape)

        attacker = PixelLevelTransferN(in_channels=3, out_channels=3, intensity=0.2)
        state_dict = torch.load('implementation/models/pretrained/gan_attacker.pkl')
        attacker.load_state_dict(state_dict)
        self.attacker = attacker


    def attack(self, input_data):

        images = Variable(input_data, volatile=True)
        images = (images + self.attacker(images)).data
        images = clamp_to_valid_img(images)
        return images


if __name__ == "__main__":


    gan_attacker = GANAttack(None, None)



    