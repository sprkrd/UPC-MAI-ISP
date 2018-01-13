def __init__(self, attack_shape):
    super(White_Box_Attacker, self).__init__(attack_shape)
    
  def attack(self, input_data, input_gradients):
    raise NotImplementedError
    
  def feedback(self, last_corrects):
    raise NotImplementedError