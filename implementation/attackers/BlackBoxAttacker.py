class BlackBoxAttacker(Attacker):
  
  def __init__(self, attack_shape):
    super(BlackBoxAttacker, self).__init__(attack_shape)
    
  def attack(self, input_data):
    raise NotImplementedError
    
  def feedback(self, last_corrects):
    raise NotImplementedError