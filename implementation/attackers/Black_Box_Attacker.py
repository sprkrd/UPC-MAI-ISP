class Black_Box_Attacker(Attacker):
  
  def __init__(self, attack_shape):
    super(Black_Box_Attacker, self).__init__(attack_shape)
    
  def attack(self, input_data):
    raise NotImplementedError
    
  def feedback(self, last_corrects):
    raise NotImplementedError