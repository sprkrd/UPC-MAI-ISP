import numpy as np

from attackers.Attacker import Attacker

from models.PixelLevelTransferN import PixelLevelTransferN

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

		self.attacker = PixelLevelTransferN(intensity)

		
		print('Loading pre-trained GANAttack')
		# XXX TODO


	def attack(self, input_data):

		return input_data + np.random.uniform(-self.intensity, self.intensity, self.attack_shape)


	