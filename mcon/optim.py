import numpy as np

class optimizer(object):

    def __init__(self, parameters):
        '''
        :param parameters: list(Node)
        '''
        self.reload_parameters(parameters)

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def reload_parameters(self, parameters):
        self.parameters = parameters
        self.t = 0

    def _step(self, param, idx):
        raise NotImplementedError

    def step(self):

        for idx, param in enumerate(self.parameters):
            self._step(param, idx)

        self.t += 1
'''
I learned these algorithms from torch offical website
'''
'''
More details are shown in 
https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
'''
class SGD(optimizer):

    def __init__(self, parameters, lr, monentum = 0., weight_decay = 0.):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = monentum
        self.weight_decay = weight_decay
        if self.momentum != 0.:
            self.b = [None for _ in parameters]

    def _step(self, param, idx):
        gt = param.grad
        if self.weight_decay != 0.:
            gt = gt + self.weight_decay * param.data

        if self.momentum != 0:
            if self.t > 0:
                self.b[idx] = self.momentum * self.b[idx] + (1 - self.lr) * gt
            else:
                self.b[idx] = gt + 0

            gt = self.b[idx]

        param.data = param.data - self.lr * gt

'''
More details are shown in 
https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad
'''
class AdaGrad(optimizer):

    def __init__(self, parameters, lr, lr_decay = 0.,
                 weight_decay = 0.,
                 initial_accumulator_value=0, eps=1e-10):
        super().__init__(parameters)
        self.lr = lr
        self.lr_decay = lr_decay

        self.weight_decay = weight_decay

        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps

        self.state_sum = [0. for _ in parameters]

    def _step(self, param, idx):
        gt = param.grad

        lr = self.lr / (1 + self.t * self.lr_decay)

        if self.weight_decay != 0:
            gt = gt + self.weight_decay * param.data

        self.state_sum[idx] = self.state_sum[idx] + gt**2
        ss = self.state_sum[idx]

        param.data = param.data - lr * gt / (np.sqrt(ss) + self.eps)

'''
More details are shown in 
https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
'''
class Adam(optimizer):

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999),
                 eps=1e-08, weight_decay=0):
        super().__init__(parameters)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [0. for _ in parameters]
        self.v = [0. for _ in parameters]
    def _step(self, param, idx):

        gt = param.grad
        if self.weight_decay != 0:
            gt = gt + self.weight_decay * param.data

        self.m[idx] = self.betas[0] * self.m[idx] + (1 - self.betas[0]) * gt
        self.v[idx] = self.betas[1] * self.v[idx] + (1 - self.betas[1]) * gt**2

        m_ = self.m[idx] / (1 - self.betas[0]**(self.t+1))
        v_ = self.v[idx] / (1 - self.betas[1]**(self.t+1))

        param.data = param.data - self.lr * m_ / (np.sqrt(v_) + self.eps)