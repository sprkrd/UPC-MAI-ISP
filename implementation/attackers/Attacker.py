class Attacker():
  
  def __init__(self, attack_shape):
    self.attack_shape = attack_shape
    
  def attack(self):
    raise NotImplementedError
    
  def feedback(self):
    raise NotImplementedError