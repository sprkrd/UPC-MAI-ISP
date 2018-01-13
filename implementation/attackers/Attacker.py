class Attacker():
  
  def __init__(self, attack_shape):
    self.attack_shape = input_shape
    self.start()
    
  def attack(self):
    raise NotImplementedError
    
  def feedback(self):
    raise NotImplementedError