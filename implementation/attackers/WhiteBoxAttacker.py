class WhiteBoxAttacker(Attacker):

    def __init__(self, attack_shape):
        super(WhiteBoxAttacker, self).__init__(attack_shape)
        
      def attack(self, model, input_data, target):
        raise NotImplementedError
        
      def feedback(self, last_corrects):
        raise NotImplementedError



class PGDAttack(WhiteBoxAttacker):

    def __init__(self, attack_shape, epsilon, a, k):
        super().__init__(attack_shape)

        self.epsilon = epsilon
        self.a = a
        self.k = k

    def attack(self, model, x_nat, y):
        """ Takes an input batch and adds a small perturbation """
        epsilon = self.epsilon # Maximum strength of the perturbation
        a = self.a # maximum change to the image
        k = self.k
        x = Variable(x_nat + 2*epsilon*(torch.rand(x_nat.size())-0.5), requires_grad=True)
        y = Variable(y)
        for _ in range(k):
            y_ = model(x)
            loss = F.cross_entropy(y_, y)
            loss.backward()
            x = x + a*torch.sign(x.grad)
            x = torch.min(torch.max(x.data, x_nat-epsilon), x_nat+epsilon)
    #         x = torch.clamp(x, -mean/std, (image_range-mean)/std)
            x = Variable(x, requires_grad=True)
        return x.detach(), y
