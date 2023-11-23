import torch
import numpy as np

def ternary_weights(initialization_way, kesi, W):   
    init = np.zeros(len(W.flatten()))
    if initialization_way == 'ternary':
        init[:round(1 / 2 * (1 - kesi) * init.size)] = 1 / np.sqrt(1 - kesi)
        init[round(1 / 2 * (1 - kesi) * init.size):2 *round(1 / 2 * (1 - kesi) * init.size)] = -1 / np.sqrt(1 - kesi)
        # c = Counter(init)
        np.random.shuffle(init)
    ter_W = torch.tensor(init.reshape(W.shape)).float()
    return ter_W


def act0(x, s1, s2, a):
    if (x < s1 or x > s2):
        y = a
    else:
        y = 0
    return y

def act1(x, r1, r2, r3, r4, b1, b2):
    if (x<r1 or x>r4):
        y = b1
    elif (x>=r2 and x<=r3):
        y = b2
    else:
        y = 0
    return y 

def sigma_torch(x, a):
    '''
    define sigma function
    '''
    return torch.heaviside(x - a, torch.tensor(0.0))

def binary0(s1, s2, b1):
    return lambda x: b1 * (sigma_torch(x, s2) + sigma_torch(-x, -s1))


def binary1(s1, s2, s3, s4, b1, b2):
    return lambda x: b1 * (sigma_torch(-x, -s1) + sigma_torch(x, s4)) + b2 * (((sigma_torch(-x, -s3) + sigma_torch(x, s2)) - 1))

def pre_acc(y):
    if y > 0:
        y = 1
    else:
        y = -1
    return y

class new_act(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, *arg):
        ctx.save_for_backward(x)
        l = len(arg[0])
        if l == 3:
            act = binary0(arg[0][0], arg[0][1], arg[0][2])
        else:
            act = binary1(arg[0][0], arg[0][1], arg[0][2], arg[0][3], arg[0][4], arg[0][5])
        # aa = self.act
        return act(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = 1*grad_output.clone()
        # grad_x = 1*
        return grad_x, None

if __name__ == '__main__':
    print(act0(1,-1, 1,3))